from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, move2nii, parse_filename, ActiDepFile
from actiDep.utils.tools import del_key, upt_dict
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants


def antsRegistration(moving, fixed, outprefix='registered', **kwargs):
    """
    Calls the ANTs registration script on the given subject.

    Parameters
    ----------
    moving : str
        Path to the moving image
    fixed : str
        Path to the fixed image
    outprefix : str
        Prefix for the output files
    kwargs : dict
        Additional arguments to pass to the registration script
    """

    # Generate temporary folder to work in
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    # Preprocess diffusion data
    print("Registering images")
    config, tools = set_config()

    regCommand = ["antsRegistrationSyNQuick.sh", "-d",
                  "3", "-f", fixed, "-m", moving, "-o", outprefix]

    for key, value in kwargs.items():
        regCommand = regCommand + [f"-{key}", value]

    call(regCommand)

    print("Registration finished")

    return temp_dir


def writeTransformSerie(moving_bidsfile, ref_bidsfile, transform_list, inverse_list=None):
    """
    Write a series of transformations to a single file.
    """

    # Using animaTransformSerieXmlGenerator
    output = transform_list[0].split('.')[0] + '.xml'

    # Get filename instead of full path
    moving_image = moving_bidsfile.filename
    ref_image = ref_bidsfile.filename
    transform_list = [os.path.basename(transform)
                      for transform in transform_list]
    if inverse_list is not None:
        inverse_list = [os.path.basename(inverse) for inverse in inverse_list]

    os.chdir(os.path.dirname(output))

    command = ['animaTransformSerieXmlGenerator', '-D']
    for transform in transform_list:
        command += ['-i', transform]
    # if inverse_list is not None:
    #     for inverse in inverse_list:
    #         command += ['-I',inverse]

    command += ['-o', output]

    print(' '.join(command))
    call(command)

    # Using a json file (BIDS compliant)
    output = output.replace('.xml', '.json')
    transform_dict = {}
    transform_dict['TransformList'] = transform_list
    transform_dict['InverseTransformList'] = inverse_list
    transform_dict['Reference'] = {'name': ref_image}
    transform_dict['Reference']['entities'] = ref_bidsfile.get_full_entities()
    transform_dict['Moving'] = {'name': moving_image}
    transform_dict['Moving']['entities'] = moving_bidsfile.get_full_entities()
    with open(output, 'w') as f:
        json.dump(transform_dict, f, indent=4)

def get_transform_list(transform_file):
    """
    Get the list of transformations from a file.

    Parameters
    ----------
    transform_file : ActiDepFile
    """

    with open(transform_file.path) as f:
        transform_dict = json.load(f)
    
    transform_list = []
    associated_files = {f.filename:f for f in transform_file.get_associations()}

    for transform in transform_dict['TransformList']:
        transform_list.append(associated_files[transform])
    
    return transform_list
    

def apply_transforms(moving, trans_list, ref, **kwargs):
    """
    Apply a list of transformations to the moving image using ANTs.

    Parameters
    ----------
    moving : ActiDepFile
        The moving image to transform
    trans_list : list of ActiDepFile
        List of transformation files to apply
    ref : ActiDepFile
        The reference image to use for the transformation
    kwargs : dict
        Additional arguments to pass to the transformation

    Returns
    -------
    str
        Path to the transformed image
    """
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    outfile=opj(temp_dir,'Warped.nii.gz')
    print("Applying transformations")

    seg_suffixes = ['mask','dseg','probseg']
    # Read moving and fixed images
    moving_image = ants.image_read(moving.path, dimension=3, pixeltype='unsigned int' if moving.suffix in seg_suffixes else 'double')
    ref_image = ants.image_read(ref.path, dimension=3, pixeltype='unsigned int' if ref.suffix in seg_suffixes else 'double')
    # Apply transforms in sequence
    transformed = ants.apply_transforms(
        fixed=ref_image,
        moving=moving_image,
        transformlist=[f.path for f in trans_list[::-1]],
        interpolator=kwargs.pop('interpolator', 'genericLabel' if moving.suffix in seg_suffixes else 'linear'),
        verbose=True,
        dimension=3,
    )

    # Save the result
    ants.image_write(transformed, outfile)

    return outfile

def registerT1onB0(subject, pipeline='msmt_csd'):
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    if isinstance(subject, str):
        subject = Subject(subject)

    print("Register B0 on T1")
    config, tools = set_config()

    dwi = subject.get(pipeline='anima_preproc',
                      desc='b0ref', extension='nii.gz')
    t1 = subject.get(suffix='T1w', scope='raw', extension='nii.gz')

    # Check that dwi and t1 have exactly one file
    if len(dwi) != 1:
        raise ValueError(
            f"Found {len(dwi)} B0 files for subject {subject.bids_id}")
    if len(t1) != 1:
        raise ValueError(
            f"Found {len(t1)} T1w files for subject {subject.bids_id}")

    dwi = dwi[0]
    t1 = t1[0]

    reg_dir = antsRegistration(moving=t1.path, fixed=dwi.path, outprefix='')
    # reg_dir = "/tmp/tmp1s2r_zkf"
    affine = os.path.join(reg_dir, '0GenericAffine.mat')
    warp = os.path.join(reg_dir, '1Warp.nii.gz')
    inverse_warp = os.path.join(reg_dir, '1InverseWarp.nii.gz')
    warped = os.path.join(reg_dir, 'Warped.nii.gz')

    entities = {'datatype': 'anat', 'pipeline': pipeline,
                'from': 'T1w', 'to': 'B0', 'suffix': 'xfm'}

    affine_target = subject.build_path(original_name='affine.mat', **entities)
    warp_target = subject.build_path(original_name='warp.nii.gz', **entities)
    warped_target = subject.build_path(
        original_name='warped.nii.gz', pipeline=pipeline, space='B0', suffix='T1w', datatype='anat')
    inverse_warp_target = subject.build_path(
        original_name='inverse_warp.nii.gz', **upt_dict(entities, {'from': 'B0', 'to': 'T1w'}))

    move2nii(affine, affine_target)
    move2nii(warp, warp_target)
    move2nii(warped, warped_target)
    move2nii(inverse_warp, inverse_warp_target)

    writeTransformSerie(moving_bidsfile=t1, ref_bidsfile=dwi, transform_list=[
                        affine_target, warp_target], inverse_list=[inverse_warp_target])


if __name__ == '__main__':
    subject = Subject('03011')
    t1 = subject.get(scope='raw', suffix='T1w', extension='nii.gz')[0]
    trans_desc = subject.get_unique(**{'from':'T1w','to':'B0'},extension='json')
    ref = subject.get_unique(pipeline='anima_preproc',desc='b0ref',extension='nii.gz')
    trans_list=get_transform_list(transform_file=trans_desc)
    moved_t1=apply_transforms(moving=t1,trans_list=trans_list,ref=ref)
    print(moved_t1)
    entities = {'datatype': 'anat', 'pipeline': 'msmt_csd',
                'suffix': 'T1w', 'space': 'B0'}
    target = subject.build_path(original_name=moved_t1, **entities)
    print(target)

    # print(apply_transforms(t1,trans_serie=trans_list,ref=ref))
