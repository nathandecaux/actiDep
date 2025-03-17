from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, move2nii, parse_filename, ActiDepFile, copy2nii
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy


def do_inverse_bvec(bvec_tmp):
    bvec = np.loadtxt(bvec_tmp)
    bvec[0, :] = -bvec[0, :]
    np.savetxt(bvec_tmp, bvec, fmt='%.18e')


def get_tissue_responses(dwi, bval, bvec, mask=None, inverse_bvec=True, **kwargs):
    """
    Calls the MRtrix3 dwi2response dhollander script to estimate the tissue responses from the DWI data.

    Parameters
    ----------
    dwi : ActiDepFile
        The DWI data to estimate the responses from
    bval : ActiDepFile
        The bval file associated with the DWI data

    bvec : ActiDepFile
        The bvec file associated with the DWI data

    mask : ActiDepFile (optional)
        Brain mask to use for the estimation

    inverse_bvec : bool (optional)
        Whether to invert the y component of the bvec file. Default is True.
    """

    # Set the config
    config, tools = set_config()
    # Set the path to the DWI data
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    dwi_tmp = copy2nii(dwi.path, opj(tmp_folder, 'dwi.nii.gz'))
    bval_tmp = copy2nii(bval.path, opj(tmp_folder, 'dwi.bval'))
    bvec_tmp = copy2nii(bvec.path, opj(tmp_folder, 'dwi.bvec'))

    if inverse_bvec:
        do_inverse_bvec(bvec_tmp)

    # Run the response estimation
    command = [
        'dwi2response',
        'dhollander',
        dwi_tmp,
        # Sortie pour WM (matière blanche)
        opj(tmp_folder, 'response_sfwm.txt'),
        opj(tmp_folder, 'response_gm.txt'),    # Sortie pour GM (matière grise)
        # Sortie pour CSF (liquide cérébrospinal)
        opj(tmp_folder, 'response_csf.txt'),
        '-fslgrad', bvec_tmp, bval_tmp,        # Gradients au format FSL
        '-force'
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    if mask is not None:
        mask_tmp = copy2nii(mask.path, opj(tmp_folder, 'brain_mask.nii.gz'))
        command += ['-mask', mask_tmp]

    call(command)

    print('Response estimation done')

    base_entities = dwi.get_entities()
    base_entities = upt_dict(base_entities, suffix='response', extension='txt')

    res_dict = {
        'response_sfwm.txt': upt_dict(base_entities, label='WM'),
        'response_gm.txt': upt_dict(base_entities, label='GM'),
        'response_csf.txt': upt_dict(base_entities, label='CSF')
    }

    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}

    return res_dict


def get_msmt_csd(dwi, bval, bvec, csf_response, gm_response, wm_response, mask=None, inverse_bvec=True, **kwargs):
    """
    Calls the MRtrix3 dwi2fod msmt_csd script to estimate the fiber orientation distributions from the DWI data.

    Parameters
    ----------
    dwi : ActiDepFile
        The DWI data to estimate the responses from
    bval : ActiDepFile
        The bval file associated with the DWI data
    bvec : ActiDepFile
        The bvec file associated with the DWI data
    csf_response : ActiDepFile
        The CSF response file
    gm_response : ActiDepFile
        The GM response file
    wm_response : ActiDepFile
        The WM response file
    mask : ActiDepFile (optional)
        Brain mask to use for the estimation
    inverse_bvec : bool (optional)
        Whether to invert the y component of the bvec file. Default is True.
    """

    # Set the config
    config, tools = set_config()
    # Set the path to the DWI data
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    dwi_tmp = copy2nii(dwi.path, opj(tmp_folder, 'dwi.nii.gz'))
    bval_tmp = copy2nii(bval.path, opj(tmp_folder, 'dwi.bval'))
    bvec_tmp = copy2nii(bvec.path, opj(tmp_folder, 'dwi.bvec'))
    wm_tmp = copy2nii(wm_response.path, opj(tmp_folder, 'wm_response.txt'))
    gm_tmp = copy2nii(gm_response.path, opj(tmp_folder, 'gm_response.txt'))
    csf_tmp = copy2nii(csf_response.path, opj(tmp_folder, 'csf_response.txt'))

    if inverse_bvec:
        do_inverse_bvec(bvec_tmp)

    # Run the response estimation
    command = [
        'dwi2fod',
        'msmt_csd',
        dwi_tmp,
        '-fslgrad', bvec_tmp, bval_tmp,        # Gradients au format FSL
        '-force'
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    if mask is not None:
        mask_tmp = copy2nii(mask.path, opj(tmp_folder, 'brain_mask.nii.gz'))
        command += ['-mask', mask_tmp]

    for i, response in enumerate(['csf', 'gm', 'wm']):
        command += [opj(tmp_folder, f'{response}_response.txt'),
                    opj(tmp_folder, f'{response}_fod.nii.gz')]

    call(command)

    print('MSMT-CSD done')

    base_entities = dwi.get_entities()
    base_entities = upt_dict(base_entities, model='fod', extension='nii.gz')

    res_dict = {
        'csf_fod.nii.gz': upt_dict(base_entities, label='CSF'),
        'gm_fod.nii.gz': upt_dict(base_entities, label='GM'),
        'wm_fod.nii.gz': upt_dict(base_entities, label='WM')
    }

    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}

    return res_dict

def normalize_fod(wmfod, gmfod, csffod, mask, **kwargs):
    """
    Run the MRtrix3 mtnormalise script to normalize the fod data.
    """

    # Set the config
    config, tools = set_config()
    # Set the path to the DWI data
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    wmfod_tmp = copy2nii(wmfod.path, opj(tmp_folder, 'wmfod.nii.gz'))
    gmfod_tmp = copy2nii(gmfod.path, opj(tmp_folder, 'gmfod.nii.gz'))
    csffod_tmp = copy2nii(csffod.path, opj(tmp_folder, 'csffod.nii.gz'))
    mask_tmp = copy2nii(mask.path, opj(tmp_folder, 'mask.nii.gz'))

    # Run the response estimation
    command = [
        'mtnormalise',
        wmfod_tmp, wmfod_tmp.replace('wmfod', 'norm_wmfod'),
        gmfod_tmp, gmfod_tmp.replace('gmfod', 'norm_gmfod'),
        csffod_tmp, csffod_tmp.replace('csffod', 'norm_csffod'),
        '-mask', mask_tmp,
        '-force'
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    call(command)

    print('FOD normalization done')

    base_entities = wmfod.get_entities()
    base_entities = upt_dict(base_entities, suffix='fod', extension='nii.gz', desc='normalized')

    res_dict = {
        'norm_wmfod.nii.gz': upt_dict(base_entities, label='WM'),
        'norm_gmfod.nii.gz': upt_dict(base_entities, label='GM'),
        'norm_csffod.nii.gz': upt_dict(base_entities, label='CSF'),
    }

    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}

    return res_dict

def fod_to_fixels(fod, mask=None, **kwargs):
    """
    Calls the MRtrix3 fod2fixel script to estimate the fixels from the fod data.

    Parameters
    ----------
    fod : ActiDepFile
        The fod data to estimate the fixels from

    """

    # Set the config
    config, tools = set_config()

    # Set the path to the DWI data
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    fod_tmp = copy2nii(fod.path, opj(tmp_folder, 'fod.nii.gz'))

    # Run the response estimation
    command = [
        'fod2fixel',
        '-force',
        fod_tmp,
        tmp_folder+'/fixels',
        '-nii'
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    if mask is not None:
        mask_tmp = copy2nii(mask.path, opj(tmp_folder, 'mask.nii.gz'))
        command += ['-mask', mask_tmp]
    
    call(command)

    print('Fixels estimation done')

    base_entities = fod.get_entities()

    res_dict = {'fixels/directions.nii': upt_dict(base_entities, suffix='fixels', extension='nii.gz', desc='directions'),
                'fixels/index.nii': upt_dict(base_entities, suffix='fixels', extension='nii.gz', desc='index')
                }
    
    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}
    return res_dict

        

def get_peaks(fod, mask=None, **kwargs):
    """
    Calls the MRtrix3 sh2peaks script to estimate the peaks from the fod data.

    Parameters
    ----------
    fod : ActiDepFile
        The fod data to estimate the peaks from

    """

    # Set the config
    config, tools = set_config()
    # Set the path to the DWI data
    tmp_folder = tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    fod_tmp = copy2nii(fod.path, opj(tmp_folder, 'fod.nii.gz'))

    # Run the response estimation
    command = [
        'sh2peaks',
        '-force',
        fod_tmp
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    command += [opj(tmp_folder, 'peaks.nii.gz')]

    call(command)

    print('Peaks estimation done')

    base_entities = fod.get_entities()
    base_entities = upt_dict(base_entities, suffix='peaks', extension='nii.gz')

    res_dict = {
        'peaks.nii.gz': base_entities
    }

    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}

    return res_dict


def filter_peaks(peaks,threshold=0.5):
    """
    Filter the peaks to keep only the ones with an amplitude above 0.1
    """
    peaks_img = sitk.ReadImage(peaks)
    peaks_arr = sitk.GetArrayFromImage(peaks_img)
    peaks_arr[peaks_arr < threshold] = 0
    peaks_img = sitk.GetImageFromArray(peaks_arr)
    peaks_img.CopyInformation(peaks_img)
    return peaks_img


if __name__ == "__main__":
    config, tools = set_config()
    subject = Subject('03011')
    # dwi = subject.get_unique(suffix='dwi', desc='preproc',
    #                          pipeline='anima_preproc', extension='nii.gz')
    # bval = subject.get_unique(extension='bval')
    # bvec = subject.get_unique(extension='bvec', desc='preproc')
    # mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')

    # pprint(get_tissue_responses(dwi, bval, bvec, mask, 'test'))

    fod = subject.get_unique(model='fod', pipeline='msmt_csd', label='WM',suffix='dwi')
    print(fod)

    # peaks = subject.get_unique(suffix='peaks', label='WM', desc='preproc', pipeline='msmt_csd', extension='nii.gz')
    # print(peaks)
    # threshold = 0.5
    # peaks_filtered = filter_peaks(peaks,threshold=threshold)
    # entities = peaks.get_entities()
    # entities = upt_dict(entities, suffix='peaks', extension='nii.gz', desc='filtered')

    # subject.write_object(peaks_filtered, **entities)