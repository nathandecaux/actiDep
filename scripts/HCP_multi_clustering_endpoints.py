import os
import json
from glob import glob
import numpy as np
from tqdm import tqdm

import vtk
from vtk.util.numpy_support import numpy_to_vtk

from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.segment.clustering import QuickBundles

from shapely.geometry import LineString
import shapely

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

# ⬇️ UMAP (pip install umap-learn)
try:
    import umap
except Exception as e:
    umap = None
    print("⚠️ UMAP non disponible (pip install umap-learn) — les modes UMAP_KMEANS seront désactivés.")


# ====================== I/O ======================

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
        pts = []
        for j in range(id_list.GetNumberOfIds()):
            pid = id_list.GetId(j)
            pts.append(polydata.GetPoint(pid))
        streamlines.append(np.array(pts))
    return streamlines


# ====================== Orientation (optionnelle) ======================

def compute_mean_streamline_quick(sl):
    qb = QuickBundles(threshold=1000.0)
    clusters = qb.cluster(sl)
    return clusters.centroids[0]

def reorient_to_reference_manhattan(streamlines, reference):
    if reference is None:
        return Streamlines(streamlines), [False]*len(streamlines)
    oriented, flips = [], []
    ref = np.asarray(reference)
    for s in streamlines:
        s = np.asarray(s)
        d_orig = np.sum(np.abs(s - ref))
        s_rev = s[::-1]
        d_flip = np.sum(np.abs(s_rev - ref))
        flip = d_flip < d_orig
        oriented.append(s_rev if flip else s)
        flips.append(flip)
    return Streamlines(oriented), flips

def reorient_from_indices(streamlines, flips):
    out = []
    for s, f in zip(streamlines, flips):
        out.append(s[::-1] if f else s)
    return Streamlines(out)


# ====================== Utils ======================

def frechet_distance(s1, s2):
    return shapely.frechet_distance(LineString(np.asarray(s1)), LineString(np.asarray(s2)))

def endpoints_from_streamlines(streamlines):
    """Retourne starts (N,3), ends (N,3)."""
    N = len(streamlines)
    starts = np.zeros((N, 3), dtype=float)
    ends   = np.zeros((N, 3), dtype=float)
    for i, s in enumerate(streamlines):
        starts[i] = s[0]
        ends[i]   = s[-1]
    return starts, ends

def cluster_all_endpoints_unified(starts, ends):
    """
    Nouvelle approche : 
    1. Concatène tous les endpoints (starts + ends) dans un seul pool
    2. Clusterise ce pool global
    3. Pour chaque streamline, récupère le label de son start et de son end
    4. Crée un label de streamline basé sur la paire (label_start, label_end)
    """
    N = len(starts)
    # Pool de tous les endpoints : shape (2*N, 3)
    all_endpoints = np.vstack([starts, ends])
    
    algo = ALGO.upper()
    
    # Clustering sur le pool complet
    if algo == "HDBSCAN":
        endpoint_labels = HDBSCAN(
            cluster_selection_epsilon=DBSCAN_EPS_MM_START, 
            min_samples=DBSCAN_MIN_SAMPLES
        ).fit_predict(all_endpoints)
    
    elif algo == "KMEANS":
        endpoint_labels = KMeans(
            n_clusters=KMEANS_K_START,  # On utilise un seul K pour tous les endpoints
            n_init=10, 
            random_state=0
        ).fit_predict(all_endpoints)
    
    elif algo == "SPECTRAL":
        endpoint_labels = spectral_labels_points(
            all_endpoints, 
            n_clusters=SPECTRAL_K_START,
            sigma=SPECTRAL_SIGMA_MM, 
            k_nn=SPECTRAL_KNN, 
            assign_labels=SPECTRAL_ASSIGN
        )
    
    elif algo == "HIERARCHICAL":
        endpoint_labels = AgglomerativeClustering(
            n_clusters=HIER_K_START, 
            linkage=HIER_LINKAGE
        ).fit_predict(all_endpoints)
    
    elif algo == "UMAP_KMEANS":
        if umap is None:
            raise RuntimeError("UMAP indisponible. Installe umap-learn.")
        reducer = umap.UMAP(
            n_components=UMAP_DIM, 
            n_neighbors=UMAP_N_NEIGH, 
            min_dist=UMAP_MIN_DIST, 
            metric=UMAP_METRIC, 
            random_state=0
        )
        emb = reducer.fit_transform(all_endpoints)
        endpoint_labels = KMeans(
            n_clusters=UMAP_K_START, 
            n_init=20, 
            random_state=0
        ).fit_predict(emb)
    
    else:
        raise ValueError(f"ALGO inconnu: {ALGO}")
    
    # Séparer les labels : premiers N pour starts, derniers N pour ends
    start_labels = endpoint_labels[:N]
    end_labels = endpoint_labels[N:]
    
    # Créer un label unique par paire (start_label, end_label)
    pairs = list(zip(start_labels, end_labels))
    uniq = {}
    streamline_labels = np.zeros(N, dtype=int)
    next_id = 0
    
    for i, pair in enumerate(pairs):
        if pair not in uniq:
            uniq[pair] = next_id
            next_id += 1
        streamline_labels[i] = uniq[pair]
    
    n_endpoint_clusters = len(set(endpoint_labels)) - (1 if -1 in endpoint_labels else 0)
    print(f"  → Clusters d'endpoints: {n_endpoint_clusters}")
    print(f"  → Paires uniques (start,end): {len(uniq)}")
    
    return streamline_labels, start_labels, end_labels, uniq

def representative_streamline_medoids_6d(streamlines, labels):
    """
    Médoïde 6D (endpoints concaténés) par cluster de fibres.
    Ignore -1 (bruit) pour les centroids exportés.
    """
    starts, ends = endpoints_from_streamlines(streamlines)
    X6 = np.hstack([starts, ends])  # (N,6)
    uniq = sorted(set(labels))
    reps_idx = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        if c == -1 or len(idx) == 0:
            reps_idx.append(None); continue
        if len(idx) == 1:
            reps_idx.append(idx[0]); continue
        D = pairwise_distances(X6[idx], metric='euclidean')
        sums = D.sum(axis=1)
        reps_idx.append(idx[np.argmin(sums)])
    return reps_idx

def write_centroids_vtk(path, centroids_streamlines):
    centroid_polydata = vtk.vtkPolyData()
    centroid_points = vtk.vtkPoints()
    centroid_lines = vtk.vtkCellArray()
    centroid_polydata.SetPoints(centroid_points)
    centroid_indices = []
    for cid, c in enumerate(centroids_streamlines):
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
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(centroid_polydata)
    w.Write()

def write_model_with_labels_vtk(path, streamlines, labels):
    model_polydata = vtk.vtkPolyData()
    model_points = vtk.vtkPoints()
    model_lines = vtk.vtkCellArray()
    model_polydata.SetPoints(model_points)
    model_centroid_indices = []
    model_point_indices = []
    for sidx, s in enumerate(streamlines):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(s))
        for i, p in enumerate(s):
            pid = model_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            line.GetPointIds().SetId(i, pid)
            model_centroid_indices.append(labels[sidx])
            model_point_indices.append(i)
        model_lines.InsertNextCell(line)
    model_polydata.SetLines(model_lines)
    arr_centroid_index = numpy_to_vtk(np.array(model_centroid_indices), deep=True)
    arr_centroid_index.SetName('centroid_index')
    model_polydata.GetPointData().AddArray(arr_centroid_index)
    arr_point_index = numpy_to_vtk(np.array(model_point_indices), deep=True)
    arr_point_index.SetName('point_index')
    model_polydata.GetPointData().AddArray(arr_point_index)
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(model_polydata)
    w.Write()


# ====================== Paramètres ======================

OUTPUT_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/flipped/centroids_endpoints"
BUNDLES = [b for b in glob('/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/vtk/summed_*.vtk')]

RESAMPLE_POINTS = 12
ORIENT = True

# Choix du mode de clustering (structure du label)
CLUSTER_MODE = "PAIR"   # "PAIR" (labels = (start,end)) ou "6D" (un seul label par fibre)

# Choix de l'algorithme
ALGO = "HDBSCAN"       # "HDBSCAN" | "KMEANS" | "SPECTRAL" | "HIERARCHICAL" | "UMAP_KMEANS"

# --- HDBSCAN ---
DBSCAN_EPS_MM_START = 1.0
DBSCAN_EPS_MM_END   = 1.0
DBSCAN_MIN_SAMPLES  = 15

# --- KMeans ---
KMEANS_K_START = 3
KMEANS_K_END   = 3
KMEANS_K_6D    = 3

# --- Spectral ---
SPECTRAL_K_START  = 2
SPECTRAL_K_END    = 2
SPECTRAL_K_6D     = 3
SPECTRAL_SIGMA_MM = 10.0
SPECTRAL_KNN      = 20
SPECTRAL_ASSIGN   = "kmeans"  # "kmeans" | "discretize"

# --- Hierarchical (Agglomerative) ---
HIER_K_START = 2
HIER_K_END   = 2
HIER_K_6D    = 4
HIER_LINKAGE = "ward"  # "ward" | "complete" | "average" | "single"
# (NB: "ward" nécessite métrique euclidienne et n_clusters>=2)

# --- UMAP + KMeans (pairs 6D) ---
UMAP_DIM       = 10
UMAP_N_NEIGH   = 30
UMAP_MIN_DIST  = 0.05
UMAP_METRIC    = "euclidean"
UMAP_K_6D      = 4         # k de KMeans après UMAP
# En mode PAIR, on appliquera UMAP+KMeans séparément sur starts et ends :
UMAP_K_START   = 3
UMAP_K_END     = 3


# ====================== Affinité Spectrale ======================

def spectral_labels_points(X, n_clusters, sigma, k_nn, assign_labels="kmeans", random_state=0):
    if X.shape[0] == 0:
        return np.array([], dtype=int)
    if X.shape[0] == 1 or n_clusters <= 1:
        return np.zeros(X.shape[0], dtype=int)
    A = kneighbors_graph(X, n_neighbors=min(k_nn, max(1, X.shape[0]-1)), mode='distance', include_self=False)
    data = A.data
    denom = max(sigma, 1e-9)
    w = np.exp(-(data * data) / (2.0 * (denom ** 2)))
    A = csr_matrix((w, A.indices, A.indptr), shape=A.shape)
    A = 0.5 * (A + A.T)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels=assign_labels,
        random_state=0,
        eigen_tol=1e-8,
        n_init=10
    )
    return sc.fit_predict(A)


# ====================== Clustering endpoints ======================

def cluster_endpoints_pair_mode(starts, ends):
    """Clusterise séparément starts et ends, puis fabrique un label par paire (cs, ce)."""
    algo = ALGO.upper()

    if algo == "HDBSCAN":
        cs = HDBSCAN(cluster_selection_epsilon=DBSCAN_EPS_MM_START, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(starts)
        ce = HDBSCAN(cluster_selection_epsilon=DBSCAN_EPS_MM_START, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(ends)

    elif algo == "KMEANS":
        cs = KMeans(n_clusters=KMEANS_K_START, n_init=10, random_state=0).fit_predict(starts)
        ce = KMeans(n_clusters=KMEANS_K_END,   n_init=10, random_state=0).fit_predict(ends)

    elif algo == "SPECTRAL":
        cs = spectral_labels_points(starts, n_clusters=SPECTRAL_K_START,
                                    sigma=SPECTRAL_SIGMA_MM, k_nn=SPECTRAL_KNN, assign_labels=SPECTRAL_ASSIGN)
        ce = spectral_labels_points(ends,   n_clusters=SPECTRAL_K_END,
                                    sigma=SPECTRAL_SIGMA_MM, k_nn=SPECTRAL_KNN, assign_labels=SPECTRAL_ASSIGN)

    elif algo == "HIERARCHICAL":
        # Agglomerative séparé pour starts et ends
        cs = AgglomerativeClustering(n_clusters=HIER_K_START, linkage=HIER_LINKAGE).fit_predict(starts)
        ce = AgglomerativeClustering(n_clusters=HIER_K_END,   linkage=HIER_LINKAGE).fit_predict(ends)

    elif algo == "UMAP_KMEANS":
        if umap is None:
            raise RuntimeError("UMAP indisponible. Installe umap-learn.")
        reducer = umap.UMAP(n_components=UMAP_DIM, n_neighbors=UMAP_N_NEIGH, min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=0)
        emb_s = reducer.fit_transform(starts)
        emb_e = reducer.fit_transform(ends)
        cs = KMeans(n_clusters=UMAP_K_START, n_init=20, random_state=0).fit_predict(emb_s)
        ce = KMeans(n_clusters=UMAP_K_END,   n_init=20, random_state=0).fit_predict(emb_e)

    else:
        raise ValueError(f"ALGO inconnu: {ALGO}")

    # Compactage (cs,ce) -> label unique 0..C-1
    pairs = list(zip(cs, ce))
    uniq = {}
    labels = np.zeros(len(pairs), dtype=int)
    next_id = 0
    for i, pr in enumerate(pairs):
        if pr not in uniq:
            uniq[pr] = next_id; next_id += 1
        labels[i] = uniq[pr]
    return labels, cs, ce, uniq

def cluster_endpoints_6d_mode(starts, ends):
    """Clusterise en 6D: [xs,ys,zs, xe,ye,ze] -> un seul label par fibre."""
    X6 = np.hstack([starts, ends])
    algo = ALGO.upper()

    if algo == "HDBSCAN":
        labels = HDBSCAN(cluster_selection_epsilon=DBSCAN_EPS_MM_START, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X6)

    elif algo == "KMEANS":
        labels = KMeans(n_clusters=KMEANS_K_6D, n_init=20, random_state=0).fit_predict(X6)

    elif algo == "SPECTRAL":
        labels = spectral_labels_points(X6, n_clusters=SPECTRAL_K_6D,
                                        sigma=SPECTRAL_SIGMA_MM, k_nn=SPECTRAL_KNN, assign_labels=SPECTRAL_ASSIGN)

    elif algo == "HIERARCHICAL":
        labels = AgglomerativeClustering(n_clusters=HIER_K_6D, linkage=HIER_LINKAGE).fit_predict(X6)

    elif algo == "UMAP_KMEANS":
        if umap is None:
            raise RuntimeError("UMAP indisponible. Installe umap-learn.")
        reducer = umap.UMAP(n_components=UMAP_DIM, n_neighbors=UMAP_N_NEIGH, min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=0)
        emb = reducer.fit_transform(X6)
        labels = KMeans(n_clusters=UMAP_K_6D, n_init=20, random_state=0).fit_predict(emb)

    else:
        raise ValueError(f"ALGO inconnu: {ALGO}")

    return labels


# ====================== Pipeline par bundle ======================

OUTPUT_DIR = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_bundle(bundle_path):
    bundle_name = os.path.basename(bundle_path).replace('.vtk', '')
    print(f"\n==================== {bundle_name} ====================")

    # 0) Lecture & (optionnel) orientation
    sl_src = load_vtk_streamlines(bundle_path)
    sl_rs  = set_number_of_points(Streamlines(sl_src), RESAMPLE_POINTS)
    if ORIENT:
        ref = compute_mean_streamline_quick(sl_rs)
        sl_rs, flips = reorient_to_reference_manhattan(sl_rs, ref)
        sl_src_oriented = reorient_from_indices(Streamlines(sl_src), flips)
    else:
        sl_src_oriented = Streamlines(sl_src)
    print(f"→ {len(sl_rs)} streamlines (rééchant.={RESAMPLE_POINTS})")

    # 1) Endpoints
    starts, ends = endpoints_from_streamlines(sl_src_oriented)

    # 2) Clustering unifié des endpoints
    print(f"Clustering endpoints (UNIFIED) avec {ALGO}…")
    labels, start_labels, end_labels, pair_map = cluster_all_endpoints_unified(starts, ends)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"→ Clusters de fibres obtenus: {n_clusters}")

    # 3) Médoïdes/centroids de fibres (6D)
    reps_idx = representative_streamline_medoids_6d(sl_rs, labels)
    centroids_streamlines = [sl_rs[i] for i in reps_idx if i is not None and labels[i] != -1]

    # 4) Exports VTK
    out_centroids = os.path.join(OUTPUT_DIR, f"{bundle_name}_centroids_endpoints_medoid.vtk")
    write_centroids_vtk(out_centroids, centroids_streamlines)

    out_model = os.path.join(OUTPUT_DIR, f"{bundle_name}_model_with_centroid_index.vtk")
    write_model_with_labels_vtk(out_model, sl_src_oriented, labels)
    print(f"✅ VTK écrits :\n  - {out_centroids}\n  - {out_model}")

    return bundle_name, int(n_clusters)


# ====================== Lancement multi-bundles ======================

if __name__ == "__main__":
    print(f"Traitement endpoints-first ({ALGO}, mode {CLUSTER_MODE}) sur {len(BUNDLES)} bundles → sortie: {OUTPUT_DIR}")
    results = []
    for b in tqdm(BUNDLES, desc="Bundles"):
        results.append(process_single_bundle(b))

    bundle_cluster_info = {name: k for (name, k) in results}
    out = os.path.join(OUTPUT_DIR, 'bundle_cluster_info.json')
    with open(out, 'w') as f:
        json.dump(bundle_cluster_info, f, indent=2)
    print(f"\n💾 Sauvegarde: {out}")
