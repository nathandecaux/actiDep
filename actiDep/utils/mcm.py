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
from actiDep.data.io import copy2nii,copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli,CLIArg
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants

def read_mcm_file(mcm_file):
    """
    Read the MCM file and return dictionary containing model structure.
    
    Parameters
    ----------
    mcm_file : str
        Path to .mcm XML file
        
    Returns
    -------
    dict
        Dictionary containing weights file and compartment information
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    
    model = {
        'weights': root.find('Weights').text,
        'compartments': []
    }
    
    for comp in root.findall('Compartment'):
        model['compartments'].append({
            'type': comp.find('Type').text,
            'filename': comp.find('FileName').text
        })
        
    return model


def mcm_estimator(dwi,bval,bvec,mask,n_comparts,**kwargs):
    """
    Calls the animaMCMEstimator to estimate the MCM coefficients from the DWI data.

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
    n_comparts : int
        Number of anisotropic compartments to estimate
    kwargs : dict  
        Additional arguments to pass to the script (eg: )
    """

    # Set the config
    config, tools = set_config()
    # Set the path to the DWI data
    tmp_folder = "/tmp/tmppk1svvog"#tempfile.mkdtemp()
    os.chdir(tmp_folder)

    # Copy the DWI data to the temporary folder
    dwi_tmp = copy2nii(dwi.path, opj(tmp_folder, 'dwi.nii.gz'))
    bval_tmp = copy2nii(bval.path, opj(tmp_folder, 'dwi.bval'))
    bvec_tmp = copy2nii(bvec.path, opj(tmp_folder, 'dwi.bvec'))
    mask_tmp = copy2nii(mask.path, opj(tmp_folder, 'mask.nii.gz'))

    command = [
        "animaMCMEstimator",
        "-b", bval_tmp,
        "-g", bvec_tmp,
        "-i", dwi_tmp,
        "-m", mask_tmp,
        "-o", "mcm.nii.gz",
        "-n", str(n_comparts)
    ]

    command = add_kwargs_to_cli(command, **kwargs)

    print(command)
    call(command)

    print('MCM estimation done')

    print(tmp_folder)
    base_entities = dwi.get_entities()
    base_entities = upt_dict(base_entities, model='MCM', extension='nii.gz')
    
    res_dict = {
        'mcm.nii.mcm': upt_dict(base_entities,extension='.mcm')
    }
    
    # Read MCM file
    mcm_model = read_mcm_file(opj(tmp_folder, 'mcm.nii.mcm'))
    
    # Add compartment files to result dictionary
    for comp in mcm_model['compartments']:
        comp_path = opj('mcm.nii', comp['filename'])
        comp_num = comp['filename'].split('_')[-1].split('.')[0]  # Get number from filename
        res_dict[comp_path] = upt_dict(base_entities, compartment=comp_num, extension='.nii.gz',desc=comp['type'].lower())
    
    res_dict['mcm.nii/mcm.nii_weights.nrrd'] = upt_dict(base_entities,extension='.nii.gz',model='MCM',desc='weights')
    
    # Add the tmp_folder to the keys
    res_dict = {opj(tmp_folder, k): v for k, v in res_dict.items()}
    return res_dict


if __name__ == "__main__":
    config,tools = set_config()
    subject = Subject('03011')
    dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc',extension='nii.gz')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')
    res_dict=mcm_estimator(dwi,bval,bvec,mask,3,R=True,F=True,S=True,c=2,ml_mode=CLIArg('ml-mode',2),opt=CLIArg('optimizer','levenberg'))
    pprint(res_dict)

    copy_from_dict(subject,res_dict,pipeline='mcm_zeppelin_test',dry_run=False)
    

