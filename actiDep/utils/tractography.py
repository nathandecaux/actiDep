from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, parse_filename, ActiDepFile
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


def rotation(in_file, out_file, angle=180, x=0, y=0, z=1):
    '''Rotate a vtp or vtk file by ${angle}° along x, y or z axis.'''
    if in_file[-3:] == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
    elif in_file[-3:] == 'vtk':
        reader = vtk.vtkPolyDataReader()
        writer = vtk.vtkPolyDataWriter()
    else:
        print("Unrecognized input file. Must be vtp or vtk.")
        return
    reader.SetFileName(in_file)
    reader.Update()
    translation = vtk.vtkTransform()
    translation.RotateWXYZ(angle, x, y, z)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()
    writer.SetFileName(out_file)
    writer.SetInputConnection(transformFilter.GetOutputPort())
    # the constant 42 is defined in IO/Legacy/vtkDataWriter.h (c++ code), it corresponds to VTK_LEGACY_READER_VERSION_4_2
    writer.SetFileVersion(42)
    writer.Update()
    writer.Write()


# command = [
#     "tckgen",
#     "-algorithm", "iFOD2",
#     "-seed_image", seeds,
#     odf_mrtrix,
#     output,
#     "-force"
# ]

def generate_ifod2_tracto(odf, seeds, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    command_args = [
        "$odf",
        "-algorithm", "iFOD2",
        "-seed_image", "$seeds",
        "-force",
        "-debug",
        "tracto.tck"
    ]

    output_patterns = {
        "tracto.tck": {
            "suffix": "tracto",
            "datatype": "dwi",
            "extension": "tck"
        }
    }
    return run_mrtrix_command('tckgen', inputs, output_patterns, entities_template=odf.get_entities(), command_args=command_args, **kwargs)

def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        print(transfo_dict)
        lps2ras = np.diag([-1, -1, 1])

        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data

def apply_affine(tracto, affine_mat, reference, target):
    """
    Apply the given affine matrix to the tractography file.

    Parameters
    ----------
    tracto : ActiDepFile
        ActiDepFile object containing the tractography file to transform.
    affine_mat : str
        Path to the affine matrix file.
    reference : str
        Path to the reference image file.
    target : str
        Path to the target image file.
    Returns
    -------
    str
        Path to the transformed tractography file.
    """

    #Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Load the affine matrix
    affine_matrix = load_matrix_in_any_format(affine_mat)
    affine_matrix = np.linalg.inv(affine_matrix)
    # Load the tractogram file
    tractogram = load_tractogram(tracto.path, reference)
    tractogram.to_space(Space.RASMM)
    # Apply the affine transformation

    tractogram.streamlines = transform_streamlines(tractogram.streamlines, affine_matrix)

    tractogram= StatefulTractogram(tractogram.streamlines, target, Space.RASMM)

    file_extension = tracto.path.split('.')[-1]
    # Save the transformed tractogram
    output_file = f'{temp_dir}/transformed_tractogram.{file_extension}'
    save_tractogram(tractogram, output_file, bbox_valid_check=False)
    return output_file

def generate_trekker_tracto(odf, seeds, n_seeds=1000, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    # odf_to_lps = run_cli_command('convert_fod', {'odf': odf}, {'odf_lps.nii.gz': odf.get_entities()}, command_args=[
    #                              '-i', odf.path, '-o', 'odf_lps.nii.gz', '-c', 'MRTRIX2ANIMA'])
    # #Get first key of odf_to_lps
    # odf_to_lps = list(odf_to_lps.keys())[0]

    command_args = [
        "track",
        odf.path,
        "--seed", seeds if isinstance(seeds, str) else seeds.path,
        "--seed_count", str(n_seeds),
        "-o", "tracto.vtk",
        "--force"
    ]

    output_patterns = {
        "tracto.vtk": {
            "suffix": "tracto",
            "datatype": "tracto",
            "extension": "vtk"
        }
    }
    result_dict = run_cli_command('trekker_linux', inputs, output_patterns, entities_template=odf.get_entities(
    ), command_args=command_args, **kwargs, use_sym_link=True)
    tract_path, tract_entities = list(result_dict.items())[0]

    # Rotate the VTK file
    rotated_path = tract_path.replace('.vtk', '_rotated.vtk')
    rotation(tract_path, rotated_path, angle=180, x=0, y=0, z=1)

    # # Convert VTK to TCK
    # call(["trekker_linux", "convert", tract_path, tract_path.replace('.vtk','.tck'), '--force'])

    result_dict[tract_path.replace('.vtk', '_rotated.vtk')] = upt_dict(tract_entities, orient='LPS')

    return result_dict


def generate_trekker_tracto_tck(odf, seeds, n_seeds=1000, **kwargs):

    inputs = {"odf": odf, "seeds": seeds}

    # odf_to_lps = run_cli_command('convert_fod', {'odf': odf}, {'odf_lps.nii.gz': odf.get_entities()}, command_args=[
    #                              '-i', odf.path, '-o', 'odf_lps.nii.gz', '-c', 'MRTRIX2ANIMA'])
    # #Get first key of odf_to_lps
    # odf_to_lps = list(odf_to_lps.keys())[0]

    command_args = [
        "track", odf.path, "--seed",
        seeds if isinstance(seeds, str) else seeds.path, "--seed_count",
        str(n_seeds), "-o", "tracto.tck", "--force"
    ]

    output_patterns = {
        "tracto.tck": {
            "suffix": "tracto",
            "datatype": "tracto",
            'algo': 'trekker',
            "extension": "tck"
        }
    }
    result_dict = run_cli_command('trekker_linux',
                                  inputs,
                                  output_patterns,
                                  entities_template=odf.get_entities(),
                                  command_args=command_args,
                                  **kwargs,
                                  use_sym_link=True)

    return result_dict


def _find_tractogram_endings(tractogram, reference):
    """
    Get the endings segmentation of the streamlines in the tractogram.
    Ensures all tracts are oriented in the same direction.

    Parameters
    ----------
    tractogram : StatefulTractogram
        The tractogram to process.
        
    reference : str
        The reference image

    Returns
    -------
    endings : dict
        Dictionary containing the start and end binary masks of the streamlines in the tractogram.
    """
    # Load reference image
    import nibabel as nib
    from dipy.tracking.streamline import orient_by_streamline
    
    ref_img = nib.load(reference)
    shape = ref_img.shape
    affine = ref_img.affine
    
    # Ensure tractogram is in the correct space
    tractogram.to_space(Space.RASMM)
    
    # Create empty binary volumes for start and end points
    start_volume = np.zeros(shape)
    end_volume = np.zeros(shape)
    
    # Get streamlines from tractogram
    streamlines = tractogram.streamlines
    
    # Check if we have valid streamlines
    if len(streamlines) == 0:
        start_img = nib.Nifti1Image(start_volume.astype('uint8'), affine)
        end_img = nib.Nifti1Image(end_volume.astype('uint8'), affine)
        return {'start': start_img, 'end': end_img}
    
    # Find the longest streamline to use as standard for orientation
    lengths = [len(s) for s in streamlines]
    standard_idx = np.argmax(lengths)
    standard = streamlines[standard_idx]
    
    # Orient all streamlines to match the standard
    oriented_streamlines = orient_by_streamline(streamlines, standard, n_points=12, in_place=False)
    
    # Get the inverse affine to convert from mm to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    # Process each oriented streamline
    for streamline in oriented_streamlines:
        if len(streamline) < 2:
            continue
            
        # Get the start and end points of the oriented streamline
        start_point = streamline[0]
        end_point = streamline[-1]
        
        # Transform points from RAS mm to voxel space
        start_voxel = np.round(nib.affines.apply_affine(inv_affine, start_point)).astype(int)
        end_voxel = np.round(nib.affines.apply_affine(inv_affine, end_point)).astype(int)
        
        # Check if points are within volume bounds and set binary mask
        if (0 <= start_voxel[0] < shape[0] and 
            0 <= start_voxel[1] < shape[1] and 
            0 <= start_voxel[2] < shape[2]):
            start_volume[start_voxel[0], start_voxel[1], start_voxel[2]] = 1
            
        if (0 <= end_voxel[0] < shape[0] and 
            0 <= end_voxel[1] < shape[1] and 
            0 <= end_voxel[2] < shape[2]):
            end_volume[end_voxel[0], end_voxel[1], end_voxel[2]] = 1
    
    # Create NIfTI images with binary masks
    start_img = nib.Nifti1Image((start_volume>0).astype('uint8'), affine)
    end_img = nib.Nifti1Image((end_volume>0).astype('uint8'), affine)
    
    return {'start': start_img, 'end': end_img}


def get_tractogram_endings(tractogram_file, reference):
    """
    Get the endings segmentation of the streamlines in the tractogram.

    Parameters
    ----------
    tractogram : ActiDepFile or str
        The tractogram to process.
        
    reference : ActiDepFile or str
        The reference image

    Returns
    -------
    result_dict : dict
        A dictionary containing the path to the generated endings segmentation files (beginning and end of each streamline).
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    tracto_path = tractogram_file if isinstance(tractogram_file, str) else tractogram_file.path
    ref_path = reference if isinstance(reference, str) else reference.path
    
    # Load the tractogram
    tractogram = load_tractogram(tracto_path, ref_path)
    
    # Get the tractogram entities if available
    if hasattr(tractogram_file, 'get_entities'):
        entities = tractogram_file.get_entities()
    else:
        # Create basic entities if not available
        filename = os.path.basename(tracto_path)
        entities = parse_filename(filename)
    
    # Get the endings segmentation
    endings = _find_tractogram_endings(tractogram, ref_path)
    
    # Save the endings segmentation
    start_file = f'{temp_dir}/streamlines_start.nii.gz'
    end_file = f'{temp_dir}/streamlines_end.nii.gz'
    
    nib.save(endings['start'], start_file)
    nib.save(endings['end'], end_file)
    
    # Create a dictionary to store the results
    result_dict = {
        start_file: upt_dict(entities, suffix='mask', label='start', desc='endings', datatype='tracto', extension='nii.gz'),
        end_file: upt_dict(entities, suffix='mask', label='end', desc='endings', datatype='tracto', extension='nii.gz')
    }
    
    return result_dict

def surface_projection(tractogram, surface, output_file, **kwargs):
    """
    Project the streamlines onto a surface.

    Parameters
    ----------
    """
    return True

def filter_tracto_by_endings(tracto, start_mask, end_mask,**kwargs):
    """
    Filter the streamlines in the tractogram based on their endpoints.

    Parameters
    ----------
    tracto : ActiDepFile or str
        The tractogram to filter.
    start_mask : ActiDepFile or str
        The binary mask defining the start region.
    end_mask : ActiDepFile or str
        The binary mask defining the end region.
    Returns
    -------
    dict
        A dictionary containing the path to the filtered tractogram file.
    """
    inputs = {
        "tracto": tracto if isinstance(tracto, str) else tracto.path,
        "start_mask": start_mask if isinstance(start_mask, str) else start_mask.path,
        "end_mask": end_mask if isinstance(end_mask, str) else end_mask.path
    }

    command_args = [
        "$tracto",
        "-include", "$start_mask",
        "-include", "$end_mask",
        "-force",
        "filtered_tracto.tck"
    ]
    entities = tracto.get_full_entities()
    output_patterns = {
        "filtered_tracto.tck": upt_dict(entities,desc='filtered',filter='endings')
    }
    return run_mrtrix_command('tckedit', inputs, output_patterns, entities_template=parse_filename(os.path.basename(tracto.path)) if hasattr(tracto, 'path') else {}, command_args=command_args, **kwargs)

def filter_tracto_by_endings_dipy(tracto, reference, start_mask, end_mask, output_file=None):
    """
    Filter the streamlines in the tractogram based on their endpoints using DIPY.

    Parameters
    ----------
    tracto : ActiDepFile or str
        The tractogram to filter.
    reference : ActiDepFile or str
        The reference image for the tractogram.
    start_mask : ActiDepFile or str
        The binary mask defining the start region.
    end_mask : ActiDepFile or str
        The binary mask defining the end region.
    output_file : str, optional
        Path to save the filtered tractogram. If None, a temporary file will be created.

    Returns
    -------
    str
        Path to the filtered tractogram file.
    """
    import nibabel as nib
    from dipy.tracking.streamline import set_number_of_points

    tracto_path = tracto if isinstance(tracto, str) else tracto.path
    ref_path = reference if isinstance(reference, str) else reference.path
    start_mask_path = start_mask if isinstance(start_mask, str) else start_mask.path
    end_mask_path = end_mask if isinstance(end_mask, str) else end_mask.path

    # Load the tractogram and masks
    tractogram = load_tractogram(tracto_path, ref_path)
    start_img = nib.load(start_mask_path)
    end_img = nib.load(end_mask_path)

    start_data = start_img.get_fdata().astype(bool)
    end_data = end_img.get_fdata().astype(bool)

    # Get the affine of the reference image
    affine = nib.load(ref_path).affine
    inv_affine = np.linalg.inv(affine)

    filtered_streamlines = []
    
    for sl in tractogram.streamlines:
        if len(sl) < 2:
            continue
        
        # Get start and end points in voxel space
        start_voxel = np.round(nib.affines.apply_affine(inv_affine, sl[0])).astype(int)
        end_voxel = np.round(nib.affines.apply_affine(inv_affine, sl[-1])).astype(int)
        
        # Check if points are within bounds and in the masks
        if (0 <= start_voxel[0] < start_data.shape[0] and 
            0 <= start_voxel[1] < start_data.shape[1] and
            0 <= start_voxel[2] < start_data.shape[2] and
            0 <= end_voxel[0] < end_data.shape[0] and
            0 <= end_voxel[1] < end_data.shape[1] and
            0 <= end_voxel[2] < end_data.shape[2]):
            if start_data[start_voxel[0], start_voxel[1], start_voxel[2]] and end_data[end_voxel[0], end_voxel[1], end_voxel[2]]:
                filtered_streamlines.append(sl)

    # Create a new tractogram with the filtered streamlines
    filtered_tractogram = StatefulTractogram(filtered_streamlines, ref_path, Space.RASMM)
    if output_file is None:
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'filtered_tracto.tck')
    save_tractogram(filtered_tractogram, output_file, bbox_valid_check=False)
    
    entities = tracto.get_full_entities()
    return {output_file: upt_dict(entities, desc='filtered', filter='endings')}



if __name__ == "__main__":
    # subject = Subject("03011")
    # odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    # seeds = sub.get_unique(suffix='mask', label='WM', space='B0')

    # output_dict = generate_ifod2_tracto(odf, seeds)
    # pprint(output_dict)
    # copy_from_dict(sub, output_dict,pipeline='msmt_csd')

    # odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    tracto = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/bundle_seg/sub-03011/tracto/sub-03011_bundle-CSTleft_desc-cleaned_tracto.trk'

    ref = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/anima_preproc/sub-03011/dwi/sub-03011_metric-FA_model-DTI_dwi.nii.gz'

    print(get_tractogram_endings(tracto, ref))