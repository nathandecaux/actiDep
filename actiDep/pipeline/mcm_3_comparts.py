import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, Actidep
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.mcm import mcm_estimator, update_mcm_info_file, add_mcm_to_tracts
import tempfile
import glob
import shutil
import multiprocessing  # Ajout de multiprocessing


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

# def mcm_to_trekker_tracts(subject, pipeline, **kwargs):
#     """
#     Project MCM metrics to IFOD2 tracts.
    
#     Parameters
#     ----------
#     subject : Subject
#         The subject object containing the data
#     pipeline : str
#         The pipeline to use for processing
#     kwargs : dict
#         Additional keyword arguments for processing
#     """
#     mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
#     tracts= subject.get_unique(suffix='tracto', algo='trekker',orient='LPS',desc='normalized', extension='vtk',pipeline='msmt_csd')
#     reference = subject.get(space='B0', datatype='anat',extension='nii.gz')[0]

#     res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
#     copy_from_dict(subject, res_dict, pipeline=pipeline)
#     return True

def mcm_to_bundleseg_tracts(subject, pipeline, bundle_name, **kwargs):
    """
    Project MCM metrics to bundlesegmentation tracts.
    
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
    tracts= subject.get_unique(suffix='tracto', bundle=bundle_name, extension='trk',pipeline='bundle_seg')
    reference = subject.get(metric='FA', pipeline='anima_preproc', datatype='dwi',extension='nii.gz')[0]

    res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)
    return True

# def mcm_to_recobundle_bundle(subject,pipeline,bundle,**kwargs):
#     """
#     Project MCM metrics to Recobundle bundle.
    
#     Parameters
#     ----------
#     subject : Subject
#         The subject object containing the data
#     pipeline : str
#         The pipeline to use for processing
#     bundle : str
#         The bundle name to convert
#     kwargs : dict
#         Additional keyword arguments for processing
#     """
#     mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
#     #Remove - and _ from the bundle name
#     bundle_short = bundle.replace('-','').replace('_','')
#     tracts= subject.get_unique(suffix='tracto', pipeline='recobundle_segmentation',space='HCP105Group1Clustered',desc='normalized', extension='vtk',desc='noslr', bundle=bundle_short)
#     reference = subject.get(space='B0', datatype='anat',extension='nii.gz')[0]

#     res_dict = add_mcm_to_tracts(mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
#     copy_from_dict(subject, res_dict, pipeline=pipeline)
#     return True

def process_mcm_pipeline(subject,pipeline='mcm_tensors'):
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
        # 'mcm_estimation',
        'mcm_to_bundleseg_tracts'
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'mcm_estimation': lambda: process_mcm_estimation(subject, pipeline, R=True,c=3,n=3,F=True,
                                                         ml_mode=CLIArg(
                                                             'ml-mode', 2),
                                                          opt=CLIArg('optimizer', 'levenberg')
                                                          ),
        'mcm_to_bundleseg_tracts': lambda: mcm_to_bundleseg_tracts(subject, pipeline, bundle_name='CSTleft'),
    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()


def process_subject(sub, dataset_path, pipeline_name):
    """
    Process a single subject - worker function for multiprocessing.
    
    Parameters
    ----------
    sub : str
        Subject ID
    dataset_path : str
        Path to the BIDS dataset
    pipeline_name : str
        Pipeline name to use
    """
    # Set temporary directory if on calcarine
    if os.uname()[1] == 'calcarine':
        tempfile.tempdir = '/local/ndecaux/tmp'
    
    # Initialize subject
    ds = Actidep(dataset_path)
    subject = ds.get_subject(sub)
    
    # Check if already processed
    if len(subject.get(model='MCM', pipeline=pipeline_name)) > 0:
        print(f"Skipping subject {sub} as tractography already exists")
        return
    
    print(f"Processing subject: {sub}")
    process_mcm_pipeline(subject, pipeline=pipeline_name)
    return sub


from pprint import pprint
if __name__ == "__main__":
    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')
    if os.uname()[1] == 'calcarine':
        tempfile.tempdir = '/local/ndecaux/tmp'
    
    # Pipeline configuration
    pipeline = 'mcm_tensors_staniz'
    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'

    subject= Subject('01002',db_root=dataset_path)
    # print(subject.get(pipeline='bundle_seg',bundle='CSTleft'))
    process_mcm_pipeline(subject, pipeline='mcm_tensors_staniz')

    
    # # Set the number of processes (adjust based on your system's capabilities)
    # num_processes = 8#max(1, multiprocessing.cpu_count() - 1)  # Use all CPUs except one
    
    # config, tools = set_config()
    # print("MSMT-CSD pipeline with multiprocessing")
    # print("=====================================")
    # print('Reading dataset')
    # ds = Actidep(dataset_path)
    # print(f"Found {len(ds.subject_ids)} subjects")
    # print(f"Using {num_processes} parallel processes")
    # print("=====================================")
    
    # # Prepare arguments for multiprocessing
    # args = [(sub, dataset_path, pipeline) for sub in ds.subject_ids]
    
    # # Use multiprocessing pool to process subjects in parallel
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.starmap(process_subject, args)
    
    # # Print summary
    # processed = [r for r in results if r is not None]
    # print(f"Completed processing {len(processed)} subjects")
