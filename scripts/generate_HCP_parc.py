import os
import numpy as np
from glob import glob

import vtk
from vtk.util.numpy_support import (
    vtk_to_numpy,
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
)

from dipy.tracking.streamline import orient_by_streamline,set_number_of_points

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from skimage import measure

# -------------------------
# Configuration / fichiers
# -------------------------
base_vtk_dir = '/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas'
vtk_files = sorted(glob(os.path.join(base_vtk_dir, 'vtk', '*.vtk')))
vtk_files = [f for f in vtk_files if 'CST_left' in f]

ref_centerlines = sorted(glob(os.path.join(base_vtk_dir, 'long_central_line', '*.vtk')))
ref_centerlines = [f for f in ref_centerlines if 'CST_left' in f]

assert len(vtk_files) > 0, "Aucun fichier vtk trouvé."
assert len(ref_centerlines) > 0, "Aucun centerline de référence trouvé."

# -------------------------
# Lecture VTK optimisée
# -------------------------
def load_vtk_streamlines(vtk_file_path):
    """
    Charge un fichier vtk (PolyData) et renvoie une liste de streamlines numpy arrays (M_i, 3).
    Utilise vtk_to_numpy pour convertir les points et les listes de lignes en numpy, évitant
    des appels GetPoint() lourds en Python.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Points -> Nx3 numpy array
    vtk_pts = polydata.GetPoints().GetData()
    if vtk_pts is None:
        return []
    pts = vtk_to_numpy(vtk_pts).reshape(-1, 3)

    # Lines -> flat connectivity array [n0, id0_0, id0_1, ..., n1, id1_0, ...]
    lines = polydata.GetLines()
    if lines is None:
        return []

    if lines.GetNumberOfCells() == 0:
        return []

    conn = vtk_to_numpy(lines.GetData()).astype(np.int64)
    streamlines = []
    p = 0
    L = conn.shape[0]
    while p < L:
        n = int(conn[p])
        ids = conn[p + 1 : p + 1 + n]
        streamlines.append(pts[ids])
        p += 1 + n

    return streamlines

# test rapide
sls = load_vtk_streamlines(vtk_files[0])
print(f"Loaded {len(sls)} streamlines from {os.path.basename(vtk_files[0])}")
print([s.shape for s in sls[:10]])

# -------------------------
# Orientation
# -------------------------
ref_sl = load_vtk_streamlines(ref_centerlines[0])[0]
sls_oriented = orient_by_streamline(sls, ref_sl)
print("Orientation done:", [s.shape for s in sls_oriented[:5]])

# -------------------------
# Indices normalisés (binned)
# -------------------------
def get_normalized_index_streamlines(streamlines, n_bins=10):
    """
    Pour chaque streamline (Nx3), génère un tableau d'indices de bins (longueur N).
    Utilise numpy vectorisé.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    normalized = []
    for sl in streamlines:
        n_pts = sl.shape[0]
        if n_pts <= 1:
            normalized.append(np.zeros(n_pts, dtype=np.int32))
            continue
        t = np.linspace(0.0, 1.0, n_pts)
        labels = np.digitize(t, bins=bin_edges, right=True) - 1  # bins in [0..n_bins-1]
        labels = np.clip(labels, 0, n_bins - 1).astype(np.int32)
        normalized.append(labels)
    return normalized

sls_normalized = get_normalized_index_streamlines(sls_oriented, n_bins=10)

# -------------------------
# Sauvegarde VTK (indices normalisés) - en un seul fichier
# -------------------------
output_dir = os.path.join(base_vtk_dir, 'flipped', 'vtk_normalized_indices')
os.makedirs(output_dir, exist_ok=True)

# Concaténation de tous les points et création de la connectivité VTK via buffers numpy
all_points = np.vstack(sls_oriented) if len(sls_oriented) > 0 else np.empty((0, 3), dtype=np.float32)
num_cells = len(sls_oriented)
cell_sizes = np.array([len(sl) for sl in sls_oriented], dtype=np.int64)

# Construire le tableau de connectivité VTK : [n0, id0, id1, ..., n1, ...]
if num_cells > 0:
    total_ids = cell_sizes.sum()
    conn = np.empty(num_cells + total_ids, dtype=np.int64)
    pos = 0
    offset = 0
    for i, sl in enumerate(sls_oriented):
        n = len(sl)
        conn[pos] = n
        conn[pos + 1 : pos + 1 + n] = np.arange(offset, offset + n, dtype=np.int64)
        pos += 1 + n
        offset += n
else:
    conn = np.empty((0,), dtype=np.int64)

# Flatten normalized indices to match all_points order
all_norm_indices = np.concatenate(sls_normalized).astype(np.float32) if len(sls_normalized) > 0 else np.empty((0,), dtype=np.float32)

# Compose polydata VTK
poly = vtk.vtkPolyData()

# Points
if all_points.size > 0:
    vtk_pts_array = numpy_to_vtk(all_points, deep=True)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(vtk_pts_array)
    poly.SetPoints(vtk_pts)

# Lines (cells)
if conn.size > 0:
    id_array = numpy_to_vtkIdTypeArray(conn, deep=True)
    lines = vtk.vtkCellArray()
    lines.SetCells(num_cells, id_array)
    poly.SetLines(lines)

# Add scalar array
if all_norm_indices.size > 0:
    norm_vtk = numpy_to_vtk(all_norm_indices, deep=True)
    norm_vtk.SetName('NormalizedIndex')
    poly.GetPointData().AddArray(norm_vtk)

# Write file
bundle_name = os.path.basename(vtk_files[0]).split('.')[0].replace('summed_', '')
out_path = os.path.join(output_dir, f'normalized_{bundle_name}.vtk')
writer = vtk.vtkPolyDataWriter()
writer.SetFileName(out_path)
writer.SetInputData(poly)
writer.Write()
print(f'Saved normalized indices to {out_path}')

# -------------------------
# SVM training optimisée
# -------------------------
def train_svm_classifiers(streamlines, normalized_indices, n_bins=10):
    """
    - Concatène tous les points une seule fois.
    - Entraîne un Random Forest binaire par bin (plus rapide que SVM).
    - Retourne la liste de classifieurs (ou None si bin vide) et les bin_edges.
    Note: Random Forest ne nécessite pas de scaling.
    """
    # Concat points et labels
    all_points = np.vstack(streamlines)
    all_labels = np.concatenate(normalized_indices).astype(np.int32)

    classifiers = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        y = (all_labels == i).astype(int)
        if y.sum() == 0:
            classifiers.append(None)
            print(f"Bin {i}: empty, skipped.")
            continue
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            n_jobs=-1,  # parallélisation automatique
            random_state=42
        )
        rf.fit(all_points, y)
        classifiers.append(rf)
        print(f"Trained Random Forest for bin {i} (positives={y.sum()})")
    return classifiers, bin_edges, all_points

classifiers, bin_edges, all_points_train = train_svm_classifiers(sls_oriented, sls_normalized, n_bins=10)
print("SVM training done.")

# -------------------------
# Attribution des labels (vectorisé)
# -------------------------
# Plus besoin de scaler avec Random Forest
n_points = all_points.shape[0]
n_bins = len(classifiers)
decision_matrix = np.full((n_points, n_bins), -np.inf, dtype=np.float32)

for i, clf in enumerate(classifiers):
    if clf is None:
        continue
    # predict_proba[:, 1] donne la probabilité de la classe positive
    decision_matrix[:, i] = clf.predict_proba(all_points)[:, 1]

labels = np.argmax(decision_matrix, axis=1).astype(np.int32)

# -------------------------
# Écriture VTK des labels (un seul fichier)
# -------------------------
out_labels_path = os.path.join(output_dir, f'svm_labels_{bundle_name}.vtk')

poly_labels = vtk.vtkPolyData()

# Points
vtk_pts_array = numpy_to_vtk(all_points, deep=True)
vtk_pts = vtk.vtkPoints()
vtk_pts.SetData(vtk_pts_array)
poly_labels.SetPoints(vtk_pts)

# Lines (on réutilise conn construit plus haut)
if conn.size > 0:
    id_array = numpy_to_vtkIdTypeArray(conn, deep=True)
    lines = vtk.vtkCellArray()
    lines.SetCells(num_cells, id_array)
    poly_labels.SetLines(lines)

# Labels array
label_vtk = numpy_to_vtk(labels, deep=True)
label_vtk.SetName('SVM_Label')
poly_labels.GetPointData().AddArray(label_vtk)

writer = vtk.vtkPolyDataWriter()
writer.SetFileName(out_labels_path)
writer.SetInputData(poly_labels)
writer.Write()
print(f"Saved SVM labels to {out_labels_path}")

# # -------------------------
# # Extraction hyperplans (marching cubes) - inchangé logiquement, mais veiller mémoire
# # -------------------------
# grid_size = 50  # ajuste si nécessaire (mémoire / précision)
# margin = 5.0
# mins = all_points.min(axis=0) - margin
# maxs = all_points.max(axis=0) + margin

# xs = np.linspace(mins[0], maxs[0], grid_size)
# ys = np.linspace(mins[1], maxs[1], grid_size)
# zs = np.linspace(mins[2], maxs[2], grid_size)
# Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
# grid_points = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T

# hyperplane_dir = os.path.join(output_dir, 'svm_hyperplanes')
# os.makedirs(hyperplane_dir, exist_ok=True)

# for i, clf in enumerate(classifiers):
#     if clf is None:
#         continue
#     gp_scaled = scaler.transform(grid_points)
#     decisions = clf.decision_function(gp_scaled)
#     volume = decisions.reshape((grid_size, grid_size, grid_size))
#     try:
#         verts, faces, normals, values = measure.marching_cubes(volume, level=0.0)
#     except Exception as e:
#         print(f"marching_cubes failed for bin {i}: {e}")
#         continue

#     # Convertir verts (indices) en coordonnées mondes
#     verts_world = np.empty_like(verts)
#     for d in range(3):
#         verts_world[:, d] = mins[d] + (maxs[d] - mins[d]) * verts[:, d] / (grid_size - 1)

#     # Créer polydata VTK
#     vtk_pts = vtk.vtkPoints()
#     vtk_pts.SetData(numpy_to_vtk(verts_world, deep=True))

#     vtk_faces = vtk.vtkCellArray()
#     # faces est (M, 3)
#     # Construire un tableau vtk pour tris: [3,id0,id1,id2, 3,id..., ...]
#     n_faces = faces.shape[0]
#     face_conn = np.empty(n_faces * 4, dtype=np.int64)
#     face_conn[0::4] = 3
#     face_conn[1::4] = faces[:, 0]
#     face_conn[2::4] = faces[:, 1]
#     face_conn[3::4] = faces[:, 2]
#     face_id_array = numpy_to_vtkIdTypeArray(face_conn, deep=True)
#     vtk_faces.SetCells(n_faces, face_id_array)

#     poly_hp = vtk.vtkPolyData()
#     poly_hp.SetPoints(vtk_pts)
#     poly_hp.SetPolys(vtk_faces)

#     # Save
#     hyperplane_path = os.path.join(hyperplane_dir, f'svm_hyperplane_{bundle_name}_bin{i}.vtk')
#     w = vtk.vtkPolyDataWriter()
#     w.SetFileName(hyperplane_path)
#     w.SetInputData(poly_hp)
#     w.Write()
#     print(f"Saved hyperplane bin {i} to {hyperplane_path}")
