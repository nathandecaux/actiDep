from pprint import pprint
import os
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config, get_HCP_bundle_names
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
    dwi = subject.get_unique(suffix='dwi', desc='preproc',
                             extension='nii.gz', pipeline='anima_preproc')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    return dwi, bval, bvec, mask


# def init_pipeline(subject, pipeline, **kwargs):
#     """Initialize the MCM pipeline"""
#     create_pipeline_description(
#         pipeline,
#         layout=subject.layout,
#         **kwargs
#     )
#     return True


def process_mcm_estimation(subject, pipeline, **kwargs):
    dwi, bval, bvec, mask = get_dwi_data(subject)
    compart_map = subject.get_unique(
        suffix='density', extension='nii.gz', desc='fixels2peaks')
    res_dict = mcm_estimator(dwi, bval, bvec, mask,
                             compart_map=compart_map, **kwargs)
    mapping = copy_from_dict(subject, res_dict, pipeline=pipeline)
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

    if 'overwrite' in kwargs:
        overwrite = kwargs.pop('overwrite')
    else:
        overwrite = False
    mcm_file = subject.get_unique(extension='mcmx', pipeline=pipeline)
    reference = subject.get(
        metric='FA', pipeline='anima_preproc', datatype='dwi', extension='nii.gz')[0]

    if bundle_name == "ALL":
        bundle_list = list(get_HCP_bundle_names().keys())
    elif isinstance(bundle_name, list):
        bundle_list = bundle_name
    else:
        bundle_list = [bundle_name]

    already_done = subject.get(suffix='tracto',
                               desc='cleaned',pipeline=pipeline, extension='vtk')
    if len(already_done) > 0 and not overwrite:
        already_done = [subject.get_full_entities()['bundle'] for subject in already_done]
        print(f"Already done: {already_done}")
        bundle_list = list(set(bundle_list) - set(already_done))

        # if len(already_done) > 40:
        #     print(
        #         f"Already done: {len(already_done)} bundles. Skipping.")
        #     return True

    for bundle_name in bundle_list:
        # #Empty the temp directories
        # for f in glob.glob(os.path.join(tempfile.gettempdir(), '*')):
        #     try:
        #         os.remove(f)
        #     except Exception as e:
        #         print(f"Error removing {f}: {e}")

        tracts = subject.get(suffix='tracto', bundle=bundle_name,
                             extension='trk', pipeline='bundle_seg')
        if len(tracts) == 0:
            print(
                f"Bundle {bundle_name} not found for subject {subject}. Skipping.")
            continue

        try:
            tracts = tracts[0]
            res_dict = add_mcm_to_tracts(
                mcm_file=mcm_file, tracts=tracts, reference=reference, **kwargs)
            copy_from_dict(subject, res_dict, pipeline=pipeline)
        except Exception as e:
            print(f"Error processing bundle {bundle_name} for subject {subject}: {e}")
            continue


        #Remove all files in res_dict.keys() basedir
        try:
            for k in res_dict.keys():
                tempfile_dir = os.path.dirname(k)
                if os.path.exists(tempfile_dir):
                    for f in glob.glob(os.path.join(tempfile_dir, '*')):
                            os.remove(f)
        except Exception as e:
            print(f"Error removing temporary files: {e}")
                    
        #Delete all the objects apart from the one used in the loop
        del res_dict
        del tracts

    
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


def process_mcm_pipeline(subject, pipeline='mcm_tensors'):
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
        # 'init',
        'mcm_estimation',
        # 'mcm_to_bundleseg_tracts'
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'mcm_estimation': lambda: process_mcm_estimation(subject, pipeline, R=True, c=3, n=3, F=True,
                                                         ml_mode=CLIArg(
                                                             'ml-mode', 2),
                                                         opt=CLIArg(
                                                             'optimizer', 'levenberg')
                                                         ),
        'mcm_to_bundleseg_tracts': lambda: mcm_to_bundleseg_tracts(subject, pipeline, bundle_name='ALL',overwrite=True),
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
    # if len(subject.get(model='MCM', pipeline=pipeline_name)) > 0:
    #     print(f"Skipping subject {sub} as tractography already exists")
    #     return

    print(f"Processing subject: {sub}")
    try:
        process_mcm_pipeline(subject, pipeline=pipeline_name)
    
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        # Optionally, you can log the error or take other actions here
        return None
    
    return sub


if __name__ == "__main__":
    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')
    if os.uname()[1] == 'calcarine':
        tempfile.tempdir = '/local/ndecaux/tmp'
        os.environ['TMPDIR'] = tempfile.tempdir
        num_processes = 32

    else:
        # Tempdir on home
        tempfile.tempdir = os.path.join(os.path.expanduser('~'), 'tmp')
        os.environ['TMPDIR'] = tempfile.tempdir
        num_processes = 1

    # Pipeline configuration
    pipeline = 'mcm_tensors_staniz'
    # dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    # dataset_path = '/home/ndecaux/Code/Data/dysdiago'

    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/IRM_Cerveau_MOI/bids'
    # subject = Subject('01002', db_root=dataset_path)
    # # # print(subject.get(pipeline='bundle_seg',bundle='CSTleft'))
    # process_mcm_pipeline(subject, pipeline='mcm_tensors_staniz')

    # Set the number of processes (adjust based on your system's capabilities)



    config, tools = set_config()
    print("MSMT-CSD pipeline with multiprocessing")
    print("=====================================")
    print('Reading dataset')
    ds = Actidep(dataset_path)
    print(f"Found {len(ds.subject_ids)} subjects")
    print(f"Using {num_processes} parallel processes")
    print("=====================================")

    # Prepare arguments for multiprocessing
    args = [(sub, dataset_path, pipeline) for sub in ds.subject_ids]
    #Sort by subject ID
    args.sort(key=lambda x: x[0])

    if os.uname()[1] != 'calcarine':
        args = args[::-1]
        

    # Use multiprocessing pool to process subjects in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_subject, args)

    # Print summary
    processed = [r for r in results if r is not None]
    print(f"Completed processing {len(processed)} subjects")
