import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.mcm import mcm_estimator, update_mcm_info_file, add_mcm_to_tracts
import tempfile
import glob
import shutil


def get_dwi_data(subject):
    """Helper function to get DWI data for a subject"""
    dwi = subject.get_unique(suffix='dwi', desc='preproc', extension='nii.gz',pipeline='anima_preproc')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    return dwi, bval, bvec, mask


def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True


def process_mcm_estimation(subject, pipeline, **kwargs):
    dwi, bval, bvec, mask = get_dwi_data(subject)
    compart_map = subject.get_unique(suffix='density', extension='nii.gz',desc='fixels2peaks')
    res_dict = mcm_estimator(dwi, bval, bvec, mask, compart_map=compart_map, **kwargs)
    mapping=copy_from_dict(subject, res_dict, pipeline=pipeline)
    # update_mcm_info_file(mapping)
    return True

def mcm_to_trekker_tracts(subject, pipeline, **kwargs):
    """
    Convert MCM to IFOD2 tracts.
    
    Parameters
    ----------
    subject : Subject
        The subject object containing the data
    pipeline : str
        The pipeline to use for processing
    kwargs : dict
        Additional keyword arguments for processing
    """
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
    tracts= subject.get_unique(suffix='tracto', algo='trekker',orient='LPS',desc='normalized', extension='vtk',pipeline='msmt_csd')
    reference = subject.get(space='B0', datatype='anat',extension='nii.gz')[0]

    res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)
    return True

def process_mcm_pipeline(subject,pipeline='mcm_zeppelin_3_comparts'):
    """
    Process the MSMT-CSD pipeline on the given subject.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)

    # Define processing steps
    pipeline_list = [
        'init',
        'mcm_estimation'
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'mcm_estimation': lambda: process_mcm_estimation(subject, pipeline,R=True, Z=True,c=3,n=3,
                                                         ml_mode=CLIArg(
                                                             'ml-mode', 2),
                                                          opt=CLIArg('optimizer', 'levenberg'),
                                                          out_mose=CLIArg('out-mose', '/home/ndecaux/Data/out_mose.nii.gz')
                                                          )
    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()

from pprint import pprint
if __name__ == "__main__":
    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')
    subject = Subject('03011')
    process_mcm_pipeline(subject, pipeline='mcm_tensors')
    mcm_to_trekker_tracts(subject, pipeline='mcm_tensors')
    print('Done')
