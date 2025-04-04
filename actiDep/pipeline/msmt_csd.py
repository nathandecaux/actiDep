#!/usr/bin/env python

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
from actiDep.utils.fod import get_tissue_responses, get_msmt_csd, get_peaks, normalize_fod, fod_to_fixels, get_peak_density, fixel_to_peaks
from actiDep.utils.tractography import generate_ifod2_tracto,generate_trekker_tracto
import tempfile
import glob
import shutil
from time import sleep


def get_dwi_data(subject):
    """Helper function to get DWI data for a subject"""
    dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc', extension='nii.gz')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', space='B0')
    return dwi, bval, bvec, mask

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MSMT-CSD pipeline"""
    create_pipeline_description(
        pipeline, 
        layout=subject.layout, 
        registrationMethod='antsRegistrationSyNQuick.sh', 
        registrationParams='default', 
        exactParams={'-d': '3'},
        **kwargs
    )
    return True

def process_response(subject, dwi_data, pipeline, **kwargs):
    """Calculate tissue response functions"""
    dwi, bval, bvec, mask = dwi_data
    print(f"Processing response with mask: {mask}")
    res_dict = get_tissue_responses(dwi, bval, bvec, mask, inverse_bvec=True, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_fod(subject, dwi_data, pipeline, **kwargs):
    """Run MSMT-CSD to calculate fiber orientation distributions"""
    dwi, bval, bvec, mask = dwi_data
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
        inverse_bvec=True,
        **kwargs
    )
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_normalize(subject, pipeline, **kwargs):
    """Normalize FODs"""
    wm_fod = subject.get_unique(label='WM', suffix='fod', pipeline=pipeline, desc='preproc')
    gm_fod = subject.get_unique(label='GM', suffix='fod', pipeline=pipeline, desc='preproc')
    csf_fod = subject.get_unique(label='CSF', suffix='fod', pipeline=pipeline, desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi', space='B0')
    res_dict = normalize_fod(wm_fod, gm_fod, csf_fod, mask, **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_fixels(subject, pipeline, **kwargs):
    """Perform fixel-based analysis"""
    wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline, desc='normalized', suffix='fod')
    mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi', space='B0')
    res_dict = fod_to_fixels(fod=wm_fod, mask=mask, max_peaks=CLIArg('maxnum', 3), **kwargs)
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_peaks(subject, pipeline, **kwargs):
    """Perform peak extraction"""
    wm_fod = subject.get_unique(label='WM', model='fod', pipeline=pipeline, desc='normalized', suffix='fod')
    peaks_dict = get_peaks(wm_fod, **kwargs)
    copy_from_dict(subject, peaks_dict, pipeline=pipeline)

def process_peak_density(subject, pipeline, **kwargs):
    """Calculate peak density from peaks"""
    peaks = subject.get_unique(suffix='peaks', label='WM', desc='normalized', pipeline=pipeline)
    peak_density = get_peak_density(peaks, **kwargs)
    entities = peaks.get_entities()
    entities = upt_dict(entities, suffix='density', extension='nii.gz', pipeline=pipeline)
    subject.write_object(peak_density, **entities)

def process_fixels2peaks(subject, pipeline, **kwargs):
    """Convert fixels to peaks"""
    fixels = subject.get_unique(extension='fixel', label='WM', pipeline=pipeline)
    peaks = fixel_to_peaks(fixels, **kwargs)
    entities = upt_dict(fixels.get_entities(), suffix='peaks', extension='nii.gz', pipeline=pipeline)
    subject.write_object(peaks, **entities)

def process_fixel_density(subject, pipeline, **kwargs):
    """Calculate density from fixel peaks"""
    fixel_peaks = subject.get_unique(suffix='peaks', label='WM', pipeline=pipeline, desc='fixels2peaks')
    fixel_density = get_peak_density(fixel_peaks, **kwargs)
    entities = upt_dict(fixel_peaks.get_entities(), suffix='density', extension='nii.gz', pipeline=pipeline)
    subject.write_object(fixel_density, **entities)

def process_ifod2_tracto(subject, pipeline, **kwargs):
    """Run iFOD2 tractography"""
    odf = subject.get_unique(suffix='fod', label='WM', desc='normalized', pipeline=pipeline)
    seeds = subject.get_unique(suffix='mask', label='WM', space='B0')
    tracto = generate_ifod2_tracto(odf, seeds, **kwargs)
    copy_from_dict(subject, tracto, pipeline=pipeline,datatype='tracto',algo='ifod2')

def process_trekker_tracto(subject, pipeline, **kwargs):
    """Run Trekker tractography"""
    odf = subject.get_unique(suffix='fod', label='WM', desc='normalized', pipeline=pipeline)
    seeds = subject.get_unique(suffix='mask', label='WM', space='B0')
    tracto = generate_trekker_tracto(odf,seeds, **kwargs)
    copy_from_dict(subject, tracto, pipeline=pipeline,datatype='tracto',algo='trekker')


def process_msmt_csd(subject):
    """
    Process the MSMT-CSD pipeline on the given subject.
    
    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    pipeline = 'msmt_csd'

    # Define processing steps
    pipeline_list = [
        # 'init',
        # 'response',
        # 'fod',
        # 'normalize',
        # 'fixels',
        # 'peaks',
        # 'peak_density',
        # 'fixels2peaks',
        # "fixel_density",
        # 'ifod2_tracto',
        'trekker_tracto'
    ]
    
    # Get DWI data that will be used across multiple steps
    dwi_data = get_dwi_data(subject)
    
    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'response': lambda: process_response(subject, dwi_data, pipeline),
        'fod': lambda: process_fod(subject, dwi_data, pipeline),
        'normalize': lambda: process_normalize(subject, pipeline),
        'fixels': lambda: process_fixels(subject, pipeline),
        'peaks': lambda: process_peaks(subject, pipeline),
        'peak_density': lambda: process_peak_density(subject, pipeline),
        'fixels2peaks': lambda: process_fixels2peaks(subject, pipeline),
        'fixel_density': lambda: process_fixel_density(subject, pipeline),
        'ifod2_tracto': lambda: process_ifod2_tracto(subject, pipeline,n_streams=CLIArg('-select', 625000)),
        'trekker_tracto': lambda: process_trekker_tracto(subject, pipeline,n_seeds=500000)
    }
    
    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
    



if __name__ == "__main__":
    config, tools = set_config()
    subject = Subject('03011')
    process_msmt_csd(subject)