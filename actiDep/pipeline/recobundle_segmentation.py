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
from actiDep.utils.recobundle import register_template_to_subject, call_recobundle
from actiDep.utils.registration import apply_transforms
import tempfile
import glob
import ants
import shutil


def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

def process_whole_brain_registration(subject, pipeline, template_path, atlas_name='HCP105',**kwargs):
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='WM',desc='normalized',extension='tck',algo=None)
    print(tracto)
    whole_brain_tract = register_template_to_subject(tracto, template_path,atlas_name=atlas_name,**kwargs)
    copy_from_dict(subject, whole_brain_tract, pipeline='recobundle_segmentation')
    return True

def segment_bundle_in_atlas_space(subject, pipeline, bundle, **kwargs):
    """
    Segment a bundle from a whole brain tractography

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to segment
    """
    # Load the whole brain tractography
    whole_brain_tract = subject.get_unique(suffix='tracto', pipeline=pipeline, label='WM', extension='trk',atlas='HCP105Group1Clustered',desc='normalized')

    # Load the bundle template
    # template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/695768/tracts/{bundle}.trk"
    template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_{bundle}.trk"

    # Register the bundle template to the subject
    bundle_tract = call_recobundle(whole_brain_tract, template_path,atlas_name='HCP105Group1Clustered',bundle_name=bundle,**kwargs)

    # Save the segmented bundle
    copy_from_dict(subject, bundle_tract, pipeline=pipeline, label=bundle)
    return True

def segment_subject_recobundle(subject,pipeline='mcm_zeppelin_3_comparts'):
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
        'whole_brain_registration',
        'segment_bundle_in_atlas_space'
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'whole_brain_registration': lambda: process_whole_brain_registration(subject, pipeline, sample_streamlines=CLIArg('nb_pts',100),transform_type=CLIArg('--x0','rigid'))
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

    process_whole_brain_registration(subject, pipeline='recobundle_segmentation', template_path='/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/clusters/whole_brain.trk', atlas_name='HCP105Group1')

    # for i in range(1, 8):
    #     segment_bundle_in_atlas_space(subject, pipeline='recobundle_segmentation', bundle=f'CC_{i}')
    # segment_subject_recobundle(subject, pipeline='recobundle_segmentation')
    # bundles = os.listdir('/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/695768/tracts/')
    # for bundle in bundles:
    #     if bundle.endswith('.trk'):
    #         bundle = bundle.split('.')[0]
    #         print('Processing bundle:',bundle)
    #         segment_bundle_in_atlas_space(subject, pipeline='recobundle_segmentation', bundle=bundle)
    # print('Done')

    # t1 = subject.get(suffix='T1w', space='B0')[0]
    # print(t1)
    # affine = subject.get_unique(suffix='xfm', pipeline='recobundle_segmentation', label='WM')

    # t1_to_ref = apply_transforms(t1, [affine], ref='/data/HCP_Data/Structural_Data_Preprocessed/695768/Images/T1w_acpc_dc_restore_brain.nii.gz')


    # subject.write_object(t1_to_ref, suffix='T1w', space='695768', pipeline='recobundle_segmentation',extension='nii.gz',datatype='anat')

