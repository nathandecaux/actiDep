import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.registration import antsRegistration, registerT1onB0, apply_transforms, get_transform_list
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description
from actiDep.utils.fod import get_tissue_responses, get_msmt_csd, get_peaks,normalize_fod,fod_to_fixels,get_peak_density
import tempfile
import glob
import shutil
from time import sleep

    # print("Register T0 on atlas")
    # registerCommand = ["python3", antsRegister3DImageOnAtlas, "-a", atlas, "-t", os.path.join("Structural/T13D", subjectPrefix + "_T13D_masked.nii.gz"), "-m",  os.path.join(outputFolder, "Preprocessed_DWI", subjectDiffPrefix + "_DWI_T0.nii.gz"), "--output-folder", outputFolder, "--trsf-folder", trsfFolder, "--atlas-name", atlasName, "-n", subjectDiffPrefix]

def process_msmt_csd(subject):
    """
    Process the MSMT-CSD pipeline on the given subject.
    """
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    if isinstance(subject, str):
        subject = Subject(subject)
    pipeline='msmt_csd'

    pipeline_list = [
                    #  'init',
                    #  'response',
                    #  'fod',
                    #  'normalize',
                     'fixels',
                    #  'peaks',
                    #  'peak_density'
                     ]
    # pipeline_list = ['fixels']
    # pipeline_list = ['normalize']

    if 'init' in pipeline_list:
        create_pipeline_description(pipeline, layout=subject.layout, registrationMethod='antsRegistrationSyNQuick.sh', registrationParams='default', exactParams={'-d': '3'})
    
    # registerT1onB0(subject)

    #1. Register tissue masks on B0 - PAS BESOIN AVEC DHOLLANDER
    # files_to_move = subject.get(pipeline=pipeline,suffix='mask',datatype='anat')
    # b0 = subject.get_unique(suffix='dwi',desc='b0ref',pipeline='anima_preproc')
    # trans_file = subject.get_unique(**{'from':'T1w','to':'B0'},pipeline=pipeline,extension='json')
    # trans_list = get_transform_list(trans_file)
    # for file in files_to_move:
    #     moved_file = apply_transforms(moving=file,trans_list=trans_list,ref=b0)
    #     subject.write_object(moved_file,**upt_dict(file.get_full_entities(),space='B0',datatype='dwi'))
    dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc',extension='nii.gz')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')

    if 'response' in pipeline_list:
        sleep(1)
        #2. Run MSMT-CSD response function estimation
        dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc',extension='nii.gz')
        bval = subject.get_unique(extension='bval')
        bvec = subject.get_unique(extension='bvec', desc='preproc')
        mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')
        print(mask)

        res_dict=get_tissue_responses(dwi, bval, bvec, mask, inverse_bvec=True)
        copy_from_dict(subject, res_dict,pipeline=pipeline)
    
    if 'fod' in pipeline_list:
        sleep(1)
        #3. Run MSMT-CSD
        csf_response = subject.get_unique(label='CSF', suffix='response', pipeline=pipeline)
        gm_response = subject.get_unique(label='GM', suffix='response', pipeline=pipeline)
        wm_response = subject.get_unique(label='WM', suffix='response', pipeline=pipeline)

        res_dict = get_msmt_csd(
            dwi=dwi, 
            bval=bval, 
            bvec=bvec, 
            csf_response=csf_response, 
            gm_response=gm_response, 
            wm_response=wm_response, 
            mask=mask,
            inverse_bvec=True
        )
        copy_from_dict(subject, res_dict,pipeline=pipeline)

    if 'normalize' in pipeline_list:
        sleep(1)
        #4. Normalize FODs
        wm_fod = subject.get_unique(label='WM', suffix='fod', pipeline=pipeline, desc='preproc')
        gm_fod = subject.get_unique(label='GM', suffix='fod', pipeline=pipeline, desc='preproc')
        csf_fod = subject.get_unique(label='CSF', suffix='fod', pipeline=pipeline, desc='preproc')
        mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi',space='B0')
        res_dict = normalize_fod(wm_fod, gm_fod, csf_fod, mask)
        copy_from_dict(subject, res_dict,pipeline=pipeline)

    
    if 'fixels' in pipeline_list:
        sleep(1)
        #5.a Perform fixel-based analysis
        wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline, desc='normalized',suffix='fod')
        mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi',space='B0')
        res_dict = fod_to_fixels(wm_fod, mask)
        copy_from_dict(subject, res_dict,pipeline=pipeline)
    
    if 'peaks' in pipeline_list:
        sleep(1)
        #5.b Perform peak extraction

        wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline,desc='normalized',suffix='fod')
        peaks_dict = get_peaks(wm_fod)
        copy_from_dict(subject, peaks_dict,pipeline=pipeline)
    
    if 'peak_density' in pipeline_list:
        sleep(1)
        # Calculate peak density
        peaks = subject.get_unique(suffix='peaks', label='WM', desc='normalized', pipeline='msmt_csd')
        peak_density = get_peak_density(peaks)
        entities = peaks.get_entities()
        entities = upt_dict(entities, suffix='density', extension='nii.gz', pipeline='msmt_csd')
        subject.write_object(peak_density, **entities)
        
if __name__ == "__main__":
    config,tools = set_config()
    subject = Subject('03011')
 
    process_msmt_csd(subject)