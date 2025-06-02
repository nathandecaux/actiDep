from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.interpolation import map_coordinates
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa

from tractseg.libs import fiber_utils


#### Custom imports ####
from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config,get_HCP_bundle_names
from actiDep.data.loader import Subject, parse_filename, ActiDepFile, Actidep
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from dipy.io.stateful_tractogram import Space, StatefulTractogram,Origin
from dipy.io.streamline import save_tractogram, load_tractogram
from time import process_time
import vtk
from dipy.tracking.streamline import transform_streamlines
from scipy.io import loadmat
import nibabel as nib
import pandas as pd

def _get_length_best_orig_peak(predicted_img, orig_img, x, y, z):
    predicted = predicted_img[x, y, z, :]       # 1 peak
    orig = [orig_img[x, y, z, 0:3], orig_img[x, y, z, 3:6], orig_img[x, y, z, 6:9]]     # 3 peaks

    angle1 = abs(np.dot(predicted, orig[0]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[0]) + 1e-7))
    angle2 = abs(np.dot(predicted, orig[1]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[1]) + 1e-7))
    angle3 = abs(np.dot(predicted, orig[2]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[2]) + 1e-7))

    argmax = np.argmax([angle1, angle2, angle3])
    best_peak_len = np.linalg.norm(orig[argmax])
    return best_peak_len


def evaluate_along_streamlines(scalar_img, streamlines, nr_points, beginnings=None, dilate=0, predicted_peaks=None,
                               affine=None):
    # Runtime:
    # - default:                2.7s (test),    56s (all),      10s (test 4 bundles, 100 points)
    # - map_coordinate order 1: 1.9s (test),    26s (all),       6s (test 4 bundles, 100 points)
    # - map_coordinate order 3: 2.2s (test),    33s (all),
    # - values_from_volume:     2.5s (test),    43s (all),
    # - AFQ:                      ?s (test),     ?s (all),      85s  (test 4 bundles, 100 points)
    # => AFQ a lot slower than others

    streamlines = list(transform_streamlines(streamlines, np.linalg.inv(affine)))

    if beginnings is not None:
        for i in range(dilate):
            beginnings = binary_dilation(beginnings)
        beginnings = beginnings.astype(np.uint8)
        streamlines = fiber_utils.orient_to_same_start_region(streamlines, beginnings)

    if predicted_peaks is not None:
        # scalar img can also be orig peaks
        best_orig_peaks = fiber_utils.get_best_original_peaks(predicted_peaks, scalar_img, peak_len_thr=0.00001)
        scalar_img = np.linalg.norm(best_orig_peaks, axis=-1)

    algorithm = "distance_map"  # equal_dist | distance_map | cutting_plane | afq


    if algorithm == "equal_dist":
        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)
        ### Aggregation ###
        values_mean = np.array(values).mean(axis=1)
        values_std = np.array(values).std(axis=1)
        return values_mean, values_std


    if algorithm == "distance_map":  # cKDTree

        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)

        ### Aggregating by cKDTree approach ###
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines)
        centroids = Streamlines(clusters.centroids)
        if len(centroids) > 1:
            print("WARNING: number clusters > 1 ({})".format(len(centroids)))
        _, segment_idxs = cKDTree(centroids.get_data(), 1, copy_data=True).query(streamlines, k=1)  # (2000, 100)

        values_t = np.array(values).T  # (2000, 100)

        # If we want to take weighted mean like in AFQ:
        # weights = dsa.gaussian_weights(Streamlines(streamlines))
        # values_t = weights * values_t
        # return np.sum(values_t, 0), None

        results_dict = defaultdict(list)
        for idx, sl in enumerate(values_t):
            for jdx, seg in enumerate(sl):
                results_dict[segment_idxs[idx, jdx]].append(seg)

        if len(results_dict.keys()) < nr_points:
            print("WARNING: found less than required points. Filling up with centroid values.")
            centroid_values = map_coordinates(scalar_img, np.array([centroids[0]]).T, order=1)
            for i in range(nr_points):
                if len(results_dict[i]) == 0:
                    results_dict[i].append(np.array(centroid_values).T[0, i])

        results_mean = []
        results_std = []
        for key in sorted(results_dict.keys()):
            value = results_dict[key]
            if len(value) > 0:
                results_mean.append(np.array(value).mean())
                results_std.append(np.array(value).std())
            else:
                print("WARNING: empty segment")
                results_mean.append(0)
                results_std.append(0)
        return results_mean, results_std


    elif algorithm == "cutting_plane":
        # This will resample all streamline to have equally distant points (resulting in a different number of points
        # in each streamline). Then the "middle" of the tract will be estimated taking the middle element of the
        # centroid (estimated with QuickBundles). Then each streamline the point closest to the "middle" will be
        # calculated and points will be indexed for each streamline starting from the middle. Then averaging across
        # all streamlines will be done by taking the mean for points with same indices.

        ### Sampling ###
        streamlines = fiber_utils.resample_to_same_distance(streamlines, max_nr_points=nr_points)
        # map_coordinates does not allow streamlines with different lengths -> use values_from_volume
        values = np.array(values_from_volume(scalar_img, streamlines, affine=np.eye(4))).T

        ### Aggregating by Cutting Plane approach ###
        # Resample to all fibers having same number of points -> needed for QuickBundles
        streamlines_resamp = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines_resamp)
        centroids = Streamlines(clusters.centroids)

        # index of the middle cluster
        middle_idx = int(nr_points / 2)
        middle_point = centroids[0][middle_idx]
        # For each streamline get idx for the point which is closest to the middle
        segment_idxs = fiber_utils.get_idxs_of_closest_points(streamlines, middle_point)

        # Align along the middle and assign indices
        segment_idxs_eqlen = []
        base_idx = 1000  # use higher index to avoid negative numbers for area below middle
        for idx, sl in enumerate(streamlines):
            sl_middle_pos = segment_idxs[idx]
            before_elems = sl_middle_pos
            after_elems = len(sl) - sl_middle_pos
            # indices for one streamline e.g. [998, 999, 1000, 1001, 1002, 1003]; 1000 is middle
            r = range((base_idx - before_elems), (base_idx + after_elems))
            segment_idxs_eqlen.append(r)
        segment_idxs = segment_idxs_eqlen

        # Calcuate maximum number of indices to not result in more indices than nr_points.
        # (this could be case if one streamline is very off-center and therefore has a lot of points only on one
        # side. In this case the values too far out of this streamline will be cut off).
        max_idx = base_idx + int(nr_points / 2)
        min_idx = base_idx - int(nr_points / 2)

        # Group by segment indices
        results_dict = defaultdict(list)
        for idx, sl in enumerate(values):
            for jdx, seg in enumerate(sl):
                current_idx = segment_idxs[idx][jdx]
                if current_idx >= min_idx and current_idx < max_idx:
                    results_dict[current_idx].append(seg)

        # If values missing fill up with centroid values
        if len(results_dict.keys()) < nr_points:
            print("WARNING: found less than required points. Filling up with centroid values.")
            centroid_sl = [centroids[0]]
            centroid_sl = np.array(centroid_sl).T
            centroid_values = map_coordinates(scalar_img, centroid_sl, order=1)
            for idx, seg_idx in enumerate(range(min_idx, max_idx)):
                if len(results_dict[seg_idx]) == 0:
                    results_dict[seg_idx].append(np.array(centroid_values).T[0, idx])

        # Aggregate by mean
        results_mean = []
        results_std = []
        for key in sorted(results_dict.keys()):
            value = results_dict[key]
            if len(value) > 0:
                results_mean.append(np.array(value).mean())
                results_std.append(np.array(value).std())
            else:
                print("WARNING: empty segment")
                results_mean.append(0)
                results_std.append(0)
        return results_mean, results_std


    elif algorithm == "afq":
        ### sampling + aggregation ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        streamlines = Streamlines(streamlines)
        weights = dsa.gaussian_weights(streamlines)
        results_mean = dsa.afq_profile(scalar_img, streamlines, affine=np.eye(4), weights=weights)
        results_std = np.zeros(nr_points)
        return results_mean, results_std


def process_projection(tracto_dict, metric_dict, **kwargs):
    """
    Process the projection of the streamlines in the bundlesegmentation tractography.

    Parameters
    ----------
    tracto_dict : list of ActiDepFile
        List of tractography files to process
    ref_img : ActiDepFile
        Reference image for the projection
    endings : ActiDepFile, optional
        Endings segmentation file. If None, no endings will be used.
    kwargs : dict
        Additional keyword arguments for processing
    """


    bundle_name_dict = get_HCP_bundle_names()


    if not "nr_points" in list(kwargs.keys()):
        nr_points = 100
    else:
        nr_points = kwargs['nr_points']


    res_list = []
    # Process each tractography file
    for bundle_name,tracto in tracto_dict.items():
        # Load the reference image
        ref_img = nib.load(metric_dict[bundle_name].path)
        affine = ref_img.affine
        scalar_img = ref_img.get_fdata()
        tracto_data = nib.streamlines.load(tracto.path).streamlines
        mean, std = evaluate_along_streamlines(scalar_img,
                                               tracto_data,
                                               beginnings=None,
                                               nr_points=nr_points,
                                               affine=affine,
                                               **kwargs)

        # Remove first and last segment as those tend to be more noisy
        mean = mean[1:-1]
        std = std[1:-1]

        tractseg_bundle_name = bundle_name_dict[bundle_name]
        # Save the results
        entities= tracto.get_full_entities()
        res_dict = upt_dict(entities, tractseg_name=tractseg_bundle_name, mean=mean, std=std)
        res_list.append(res_dict)
    # Save the results to a CSV file
    res_df = pd.DataFrame(res_list)
    #Create temporary directory
    temp_dir = tempfile.mkdtemp()

    res_df.to_csv(os.path.join(temp_dir, 'projection.csv'), index=False)

    ref_entities = metric_dict[bundle_name].get_full_entities()
    del ref_entities['bundle']

    csv_path = os.path.join(temp_dir, 'mean.csv')

    # Organiser les données pour avoir une colonne par tractseg_name
    mean_data = {}
    for tract_data in res_list:
        tractseg_name = tract_data['tractseg_name']
        mean_values = tract_data['mean']

        # S'assurer que toutes les moyennes ont la même longueur
        if not mean_data:  # Premier faisceau, initialiser la structure
            # Créer un tableau de lignes (correspond aux points le long du faisceau)
            rows = np.zeros((len(mean_values), 0))
            mean_data = {
                'data': rows,
                'columns': []
            }

        # Ajouter les données pour ce faisceau comme nouvelle colonne
        mean_data['data'] = np.column_stack((mean_data['data'], mean_values))
        mean_data['columns'].append(tractseg_name)

    # Enregistrer le CSV avec une colonne par faisceau
    header = ";".join(mean_data['columns'])
    np.savetxt(csv_path, mean_data['data'], delimiter=";", header=header, comments="")

    out_dict = {
        os.path.join(temp_dir, 'projection.csv'):
        upt_dict(
            ref_entities, {
                'suffix': 'projection',
                'extension': 'csv',
                'datatype': kwargs.get('datatype', 'projection')
            }),
        os.path.join(temp_dir, 'mean.csv'):
        upt_dict(
            ref_entities, {
                'suffix': 'mean',
                'extension': 'csv',
                'datatype': kwargs.get('datatype', 'projection')
            })
    }

    return out_dict


def process_tractseg_analysis(
        subjects_txt,
        dataset_path="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids",
        with_3dplot=False,
        metric='FA'):
    """
    Process the tractseg analysis for the given subjects.

    Parameters
    ----------
    subjects_txt : str
        Path to the text file containing the list of subjects.
    """

    bundle_name_dict = get_HCP_bundle_names()

    available_bundles = [
        'AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CC_1', 'CC_2', 'CC_3',
        'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left',
        'CST_right', 'FPT_left', 'FPT_right', 'ICP_left', 'ICP_right',
        'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'OR_left',
        'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
        'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right',
        'SLF_III_left', 'SLF_III_right', 'STR_left', 'STR_right', 'UF_left',
        'UF_right', 'T_PREM_left', 'T_PREM_right', 'T_PAR_left', 'T_PAR_right',
        'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right',
        'ST_PREM_left', 'ST_PREM_right'
    ]

    # Load the subjects
    with open(subjects_txt, 'r') as f:
        subjects = [line.strip() for line in f.readlines() if line.strip()]
    tracto_path = subjects.pop(0)

    if 'bundle' in subjects[0]:
        bundle_list = subjects.pop(0)
        bundle_header, no_header_bundle_list = bundle_list.split('=')
        curated_list = [
            b.strip() for b in no_header_bundle_list.split()
            if b.strip() in available_bundles
        ]

        bundle_list = f"{bundle_header}={' '.join(curated_list)}"

        print(
            f"Excluded bundles not in available bundles: {set(no_header_bundle_list.split()) - set(available_bundles)}"
        )
    else:
        bundle_list = ''

    if 'plot_3D' in subjects[0]:
        plot_3D = subjects.pop(0)
        plot_3D_sub = plot_3D.split('/')[1]
        print(f"plot_3D_sub: {plot_3D_sub}")
    else:
        plot_3D = ''
        plot_3D_sub = ''
        with_3dplot = False
    # Parse subject information (format: "subject_id group")
    subjects_data = [line.split(' ', 1) for line in subjects]
    subjects_df = pd.DataFrame(subjects_data, columns=['subject', 'group'])

    # Skip the header row if present
    if len(subjects_df) > 0:
        subjects_df = subjects_df.iloc[1:]

    ds = Actidep(dataset_path)
    #Créer un dossier temporaire pour stocker les résultats
    temp_dir = tempfile.mkdtemp()
    tracto_path = tracto_path.replace('TEMPDIR', temp_dir)
    print(f"Temporary directory created at: {temp_dir}")

    if with_3dplot:
        plot_3D = plot_3D.replace('TEMPDIR', temp_dir)

        print(f"3D plot directory: {plot_3D}")

    for sub in subjects_df['subject']:
        print(f"Processing subject: {sub}")
        sub_id = sub.split('-')[-1]
        files = ds.get(sub_id,
                       pipeline='bundle_seg',
                       suffix='mean',
                       metric=metric,
                       extension='csv')
        if len(files) == 0:
            print(f"No files found for subject {sub_id}. Skipping.")
            continue
        else:
            tracto_metrics = files[0]
        #Copy to <temp_dir>/<sub>/Tractometry.csv
        sub_temp_dir = os.path.join(temp_dir, sub)
        os.makedirs(sub_temp_dir, exist_ok=True)
        shutil.copy(tracto_metrics.path,
                    os.path.join(sub_temp_dir, 'Tractometry.csv'))

        if with_3dplot and plot_3D_sub == sub:
            bundles = ds.get(sub_id,
                             pipeline='bundle_seg',
                             suffix='tracto',
                             extension='tck')
            brain_mask = ds.get(sub_id,
                                pipeline='anima_preproc',
                                suffix='mask',
                                label='brain')[0]
                                
            if len(bundles) == 0:
                print(
                    f"No bundles found for subject {sub_id}. Skipping 3D plot."
                )
                with_3dplot = False
                continue

            bundles = [(b.path,
                        bundle_name_dict[parse_filename(b.path).get('bundle')])
                       for b in bundles]
            for bundle, bundle_name in bundles:
                bundle_symlink = pathlib.Path(plot_3D.split(
                    '=')[-1]) / 'TOM_trackings' / f"{bundle_name}.tck"
                bundle_symlink.parent.mkdir(parents=True, exist_ok=True)

                fake_ending = pathlib.Path(
                    plot_3D.split('=')[-1]) / 'endings_segmentations' / f"{bundle_name}_b.nii.gz"
                fake_ending.parent.mkdir(parents=True, exist_ok=True)

                brain_mask_path = pathlib.Path(
                    sub_temp_dir) / "nodif_brain_mask.nii.gz"
                brain_mask_path.parent.mkdir(parents=True, exist_ok=True)


                if not bundle_symlink.exists():
                    bundle_symlink.symlink_to(bundle)
                    # Create a fake endings segmentation file
                    if not fake_ending.exists():
                        fake_ending.symlink_to(brain_mask.path)
                    # Copy the brain mask to the temp directory
                    if brain_mask_path.exists():
                        brain_mask_path.unlink()  # Supprimer le lien existant avant d'en créer un nouveau
                    brain_mask_path.symlink_to(brain_mask.path)
                    print(
                        f"Created symlink for bundle {bundle_name} at {bundle_symlink}"
                    )
                else:
                    print(
                        f"Symlink for bundle {bundle_name} already exists at {bundle_symlink}"
                    )

    #Recreate the csv file with the new path, and save in the temp directory
    with open(os.path.join(temp_dir, 'subjects.txt'), 'w') as f:
        f.write(f"{tracto_path}\n")
        f.write(f"{bundle_list}\n")
        f.write(f"{plot_3D}\n")
        for subject_line in subjects:
            f.write(f"{subject_line}\n")

    cmd = [
        'plot_tractometry_results', '-i',
        os.path.join(temp_dir, 'subjects.txt'), '-o',
        os.path.join(temp_dir, 'tractometry_results.png')
    ]

    if with_3dplot:
        cmd += ['--plot3D', 'pval', '--tracking_format', 'tck']
    print(f"Running command: {' '.join(cmd)}")
    call(cmd)
    print(
        f"Tractometry results saved to {os.path.join(temp_dir, 'tractometry_results.png')}"
    )


if __name__ == "__main__":
    process_tractseg_analysis("/home/ndecaux/Code/actiDep/subjects.txt",metric='AD',with_3dplot=True)
