import os
import json
from glob import glob
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import Feature
from dipy.segment.metric import Metric
from dipy.tracking.streamline import Streamlines, set_number_of_points
from shapely.geometry import LineString
import shapely
import pywt  # PyWavelets pour la décomposition en ondelettes
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def load_vtk_streamlines(vtk_file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()
    lines = polydata.GetLines()
    streamlines = []
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        line_points = []
        for j in range(id_list.GetNumberOfIds()):
            point_id = id_list.GetId(j)
            point = polydata.GetPoint(point_id)
            line_points.append(point)
        streamlines.append(np.array(line_points))
    return streamlines

# --- Réorientation par distance de Manhattan avec la streamline moyenne ---
def compute_mean_streamline(streamlines):
    qb = QuickBundles(threshold=1000)
    clusters = qb.cluster(streamlines)
    return clusters.centroids[0]

def reorient_to_reference_manhattan(streamlines, reference):
    if reference is None:
        return streamlines
    oriented = []
    for s in streamlines:
        if len(s) == 0:
            oriented.append(s)
            continue
        d_orig = np.sum(np.abs(s - reference))
        s_rev = s[::-1]
        d_flip = np.sum(np.abs(s_rev - reference))
        oriented.append(s_rev if d_flip < d_orig else s)
    return Streamlines(oriented)

class FrechetDistanceFeature(Feature):
    def infer_shape(self, datum):
        return np.asarray(datum).shape
    def extract(self, datum):
        return np.asarray(datum)

class FrechetDistanceMetric(Metric):
    def are_compatible(self, shape1, shape2):
        return True
    def distance(self, feature1, feature2):
        return shapely.hausdorff_distance(LineString(feature1), LineString(feature2))
    def dist(self, features1, features2):
        return self.distance(features1, features2)

# Mettre un dossier de sortie explicite pour l'approche ondelettes 3D
output_dir = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/flipped/centroids_wavelet3d"
os.makedirs(output_dir, exist_ok=True)
threshold = 23.0
corr_threshold = 0.85  # seuil d'intercorrélation pour regrouper les streamlines
# (Supprimer la config UMAP; remplacé par des features ondelettes)

def wavelet_features_3d(streamlines, wavelet='haar', level=None, normalize=True, mode='periodization'):
    """
    Features ondelettes 3D via wavedecn sur un volume (N_points, 3, 2).
    - Axe 0: progression le long de la streamline
    - Axe 1: composantes x,y,z
    - Axe 2: duplication (taille=2) pour permettre une vraie DWT 3D
    """
    feats = []
    w = pywt.Wavelet(wavelet)
    for s in streamlines:
        arr = np.asarray(s)
        if arr.ndim != 2 or arr.shape[1] != 3:
            feats.append(np.zeros(1, dtype=float))
            continue
        # Volume 3D: (N, 3, 2) en dupliquant les coordonnées sur le 3e axe
        vol = np.stack([arr, arr], axis=2)
        # Niveau max autorisé par les tailles et le filtre
        max_level = pywt.dwtn_max_level(vol.shape, w)  # correction: passer l'objet Wavelet
        lvl = max_level if level is None else min(level, max_level)
        coeffs = pywt.wavedecn(vol, wavelet=w, level=lvl, mode=mode)
        # Aplatissement déterministe: approximation puis détails triés par clés
        flat = [coeffs[0].ravel()]
        for d in coeffs[1:]:
            for key in sorted(d.keys()):
                flat.append(d[key].ravel())
        f = np.concatenate(flat, axis=0)
        if normalize:
            f = f / (np.linalg.norm(f) + 1e-8)
        feats.append(f)
    # Sécurité: pad si nécessaire (devrait être constant car N_points fixé)
    maxlen = max((len(f) for f in feats), default=0)
    feats_padded = [np.pad(f, (0, maxlen - len(f))) if len(f) < maxlen else f for f in feats]
    return np.vstack(feats_padded) if feats_padded else np.empty((0, 0))

bundle_cluster_info = {}
bundles = glob('/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/vtk/summed_*.vtk')

for bundle_path in bundles:
    if 'CC_3' not in bundle_path:
        continue

    print('\n')
    bundle_name = os.path.basename(bundle_path).replace('.vtk', '')
    print(f"---------------- Processing {bundle_name} --------------")
    sl = load_vtk_streamlines(bundle_path)
    # Rééchantillonnage
    sl = set_number_of_points(Streamlines(sl), 12)
    # Calcul de la streamline moyenne et réorientation Manhattan
    mean_ref = compute_mean_streamline(sl)
    sl = reorient_to_reference_manhattan(sl, mean_ref)

    # Statistiques longueur (optionnel)
    lengths = []
    for s in sl:
        if len(s) > 0:
            lengths.append(LineString(s).length)
    if len(lengths) > 0:
        mean_length = np.mean(lengths)
        print(f"Mean streamline length for {bundle_name}: {mean_length:.2f} mm")

    print(f"Processing {bundle_name} with 3D Wavelet features + corr-thresholding (tau={corr_threshold})")

    max_clusters = 2
    if bundle_name.replace('left', 'right') in bundle_cluster_info.keys():
        max_clusters = bundle_cluster_info[bundle_name.replace('left', 'right')]
    elif bundle_name.replace('right', 'left') in bundle_cluster_info.keys():
        max_clusters = bundle_cluster_info[bundle_name.replace('right', 'left')]

    print(f"Max clusters for {bundle_name}: {max_clusters}")
    
    # Features par ondelettes 3D
    X = wavelet_features_3d(sl, wavelet='haar', level=None, normalize=False, mode='periodization')
    if X.shape[0] == 0:
        print(f"No streamlines to cluster for {bundle_name}")
        continue

    # Intercorrélation (corrélation normalisée par streamline) et seuillage
    Z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    sim = (Z @ Z.T) / Z.shape[1]
    np.fill_diagonal(sim, 1.0)
    adj = (sim >= corr_threshold)
    adj_sparse = csr_matrix(adj)
    n_comp, labels = connected_components(adj_sparse, directed=False)
    clusters_indices = [np.where(labels == k)[0].tolist() for k in range(n_comp)]

    # Supprimer les petits clusters (<100 streamlines)
    bad_clusters = [idxs for idxs in clusters_indices if len(idxs) < 100]
    for idxs in bad_clusters:
        print(f"Removing bad cluster with {len(idxs)} streamlines for {bundle_name}")
    clusters_indices = [idxs for idxs in clusters_indices if len(idxs) >= 100]

    # Calcul des centroïdes: moyenne point-à-point des streamlines d'un cluster
    centroids = []
    for cid, idxs in enumerate(clusters_indices):
        if len(idxs) == 0:
            continue
        centroid = np.mean([sl[i] for i in idxs], axis=0)
        centroids.append(centroid)

    print(f"Bundle {bundle_name} - Number of clusters (Wavelet3D+CorrThr): {len(clusters_indices)}")
    bundle_cluster_info[bundle_name] = len(clusters_indices)

    # Ecriture des centroids en VTK
    centroid_polydata = vtk.vtkPolyData()
    centroid_points = vtk.vtkPoints()
    centroid_lines = vtk.vtkCellArray()
    centroid_polydata.SetPoints(centroid_points)
    centroid_indices = []
    for cid, c in enumerate(centroids):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(c))
        for i, p in enumerate(c):
            pid = centroid_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            line.GetPointIds().SetId(i, pid)
            centroid_indices.append(cid)
        centroid_lines.InsertNextCell(line)
    centroid_polydata.SetLines(centroid_lines)
    centroid_index_array = numpy_to_vtk(np.array(centroid_indices), deep=True)
    centroid_index_array.SetName('centroid_index')
    centroid_polydata.GetPointData().AddArray(centroid_index_array)
    centroid_writer = vtk.vtkPolyDataWriter()
    centroid_writer.SetFileName(os.path.join(output_dir, f"{bundle_name}_centroids.vtk"))
    centroid_writer.SetInputData(centroid_polydata)
    centroid_writer.Write()

    # Ecriture du modèle avec centroid_index et point_index
    model_polydata = vtk.vtkPolyData()
    model_points = vtk.vtkPoints()
    model_lines = vtk.vtkCellArray()
    model_polydata.SetPoints(model_points)

    # Mapping streamline -> cluster id
    streamline_cluster_ids = np.full(len(sl), -1, dtype=int)
    for cid, idxs in enumerate(clusters_indices):
        for sidx in idxs:
            streamline_cluster_ids[sidx] = cid

    model_centroid_indices = []
    model_point_indices = []
    for sidx, s in enumerate(sl):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(s))
        for i, p in enumerate(s):
            pid = model_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            line.GetPointIds().SetId(i, pid)
            model_centroid_indices.append(streamline_cluster_ids[sidx])
            model_point_indices.append(i)
        model_lines.InsertNextCell(line)
    model_polydata.SetLines(model_lines)
    arr_centroid_index = numpy_to_vtk(np.array(model_centroid_indices), deep=True)
    arr_centroid_index.SetName('centroid_index')
    model_polydata.GetPointData().AddArray(arr_centroid_index)
    arr_point_index = numpy_to_vtk(np.array(model_point_indices), deep=True)
    arr_point_index.SetName('point_index')
    model_polydata.GetPointData().AddArray(arr_point_index)
    model_writer = vtk.vtkPolyDataWriter()
    model_writer.SetFileName(os.path.join(output_dir, f"{bundle_name}_model_with_centroid_index.vtk"))
    model_writer.SetInputData(model_polydata)
    model_writer.Write()

json_output_path = os.path.join(output_dir, 'bundle_cluster_info.json')
with open(json_output_path, 'w') as f:
    json.dump(bundle_cluster_info, f, indent=2)
print(f"Saved bundle cluster information to {json_output_path}")