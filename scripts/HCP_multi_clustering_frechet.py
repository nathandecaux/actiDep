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
from multiprocessing import Pool, cpu_count

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
    reorient_index=[]
    for i,s in enumerate(streamlines):
        if len(s) == 0:
            oriented.append(s)
            continue
        d_orig = np.sum(np.abs(s - reference))
        s_rev = s[::-1]
        d_flip = np.sum(np.abs(s_rev - reference))
        oriented.append(s_rev if d_flip < d_orig else s)
        reorient_index.append(True if d_flip < d_orig else False)
    return Streamlines(oriented), reorient_index

def reorient_from_indices(streamlines, reorient_indices):
    reoriented = []
    for s, flip in zip(streamlines, reorient_indices):
        if flip:
            reoriented.append(s[::-1])
        else:
            reoriented.append(s)
    return Streamlines(reoriented)

class FrechetDistanceFeature(Feature):
    def infer_shape(self, datum):
        return np.asarray(datum).shape
    def extract(self, datum):
        return np.asarray(datum)

class FrechetDistanceMetric(Metric):
    def are_compatible(self, shape1, shape2):
        return True
    def distance(self, feature1, feature2):
        return shapely.frechet_distance(LineString(feature1), LineString(feature2))
    def dist(self, features1, features2):
        return self.distance(features1, features2)

class NormFrechetDistanceFeature(Feature):
    def infer_shape(self, datum):
        return np.asarray(datum).shape
    def extract(self, datum):
        return np.asarray(datum)

class NormFrechetDistanceMetric(Metric):
    def are_compatible(self, shape1, shape2):
        return True
    def distance(self, feature1, feature2):
        line1 = LineString(feature1)
        line2 = LineString(feature2)
        frechet_dist = shapely.hausdorff_distance(line1, line2)
        ending_distance = max(min(shapely.Point(line1.coords[0]).distance(shapely.Point(line2.coords[0])),shapely.Point(line1.coords[0]).distance(shapely.Point(line2.coords[-1]))),min(shapely.Point(line1.coords[-1]).distance(shapely.Point(line2.coords[0])), shapely.Point(line1.coords[-1]).distance(shapely.Point(line2.coords[-1]))))
        
        norm_factor = line1.length + line2.length
        if norm_factor < 1e-10:
            return 0.0
        return ending_distance# / norm_factor
    def dist(self, features1, features2):
        return self.distance(features1, features2)

output_dir = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/flipped/centroids_frechetlongnorm"
os.makedirs(output_dir, exist_ok=True)
threshold = 20

# feature = FrechetDistanceFeature()
# metric = FrechetDistanceMetric(feature=feature)

feature = NormFrechetDistanceFeature()
metric = NormFrechetDistanceMetric(feature=feature)

bundle_cluster_info = {}
bundles = glob('/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/vtk/summed_*.vtk')
bundles = [b for b in bundles if 'CC_3' in b]



def process_single_bundle(args):
    """Process a single bundle - designed for parallel execution"""
    bundle_path, output_dir, threshold, bundle_cluster_info_shared = args
    
    bundle_name = os.path.basename(bundle_path).replace('.vtk', '')
    print(f"\n---------------- Processing {bundle_name} --------------")
    
    sl_source = load_vtk_streamlines(bundle_path)
    # Rééchantillonnage
    sl = set_number_of_points(Streamlines(sl_source), 12)
    #Save as VTK for checking
    vtk_polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    vtk_polydata.SetPoints(points)
    for s in sl:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(s))
        for i, p in enumerate(s):
            pid = points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            line.GetPointIds().SetId(i, pid)
        lines.InsertNextCell(line)
    vtk_polydata.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(output_dir, f"{bundle_name}_resampled.vtk"))
    writer.SetInputData(vtk_polydata)
    writer.Write()
    # Calcul de la streamline moyenne et réorientation Manhattan
    mean_ref = compute_mean_streamline(sl)
    sl ,reorient_index = reorient_to_reference_manhattan(sl, mean_ref)
    sl_source = reorient_from_indices(Streamlines(sl_source), reorient_index)
    # Statistiques longueur
    lengths = []
    for s in sl:
        if len(s) > 0:
            lengths.append(LineString(s).length)
    if len(lengths) > 0:
        mean_length = np.mean(lengths)
        print(f"Mean streamline length for {bundle_name}: {mean_length:.2f} mm")

    print(f"Processing {bundle_name} with threshold {threshold} (Frechet)")

    max_clusters = 3
    if bundle_name.replace('left', 'right') in bundle_cluster_info_shared:
        max_clusters = bundle_cluster_info_shared[bundle_name.replace('left', 'right')]
    elif bundle_name.replace('right', 'left') in bundle_cluster_info_shared:
        max_clusters = bundle_cluster_info_shared[bundle_name.replace('right', 'left')]

    print(f"Max clusters for {bundle_name}: {max_clusters}")

    qb = QuickBundles(threshold=threshold, metric=metric, max_nb_clusters=max_clusters)
    clusters = qb.cluster(sl)

    # Remove small clusters
    bad_clusters = clusters.get_small_clusters(800)
    for bad_cluster in bad_clusters:
        print(f"Removing bad cluster with {len(bad_cluster.indices)} streamlines for {bundle_name}")
        clusters.remove_cluster(bad_cluster)

    centroids = clusters.centroids
    print(f"Bundle {bundle_name} - Number of clusters (Frechet): {len(clusters)}")

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
    centroid_writer.SetFileName(os.path.join(output_dir, f"{bundle_name}_qbcentroids.vtk"))
    centroid_writer.SetInputData(centroid_polydata)
    centroid_writer.Write()
    print('coucou',os.path.join(output_dir, f"{bundle_name}_qbcentroids.vtk"))
    # Ecriture du modèle avec centroid_index et point_index
    model_polydata = vtk.vtkPolyData()
    model_points = vtk.vtkPoints()
    model_lines = vtk.vtkCellArray()
    model_polydata.SetPoints(model_points)

    # Mapping streamline -> cluster id
    streamline_cluster_ids = np.full(len(sl), -1, dtype=int)
    for cid, c in enumerate(clusters):
        for sidx in c.indices:
            streamline_cluster_ids[sidx] = cid    
            
    model_centroid_indices = []
    model_point_indices = []
    for sidx, s in enumerate(sl_source):
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

    return bundle_name, len(clusters)


# Prepare arguments for parallel processing
args_list = [(bundle_path, output_dir, threshold, bundle_cluster_info) for bundle_path in bundles]

# Use multiprocessing to process bundles in parallel
n_processes = min(cpu_count() - 1, len(bundles))  # Leave one CPU free
print(f"Processing {len(bundles)} bundles using {n_processes} processes")

with Pool(processes=n_processes) as pool:
    results = pool.map(process_single_bundle, args_list)

# Collect results
for bundle_name, n_clusters in results:
    bundle_cluster_info[bundle_name] = n_clusters

json_output_path = os.path.join(output_dir, 'bundle_cluster_info.json')
with open(json_output_path, 'w') as f:
    json.dump(bundle_cluster_info, f, indent=2)
print(f"\nSaved bundle cluster information to {json_output_path}")