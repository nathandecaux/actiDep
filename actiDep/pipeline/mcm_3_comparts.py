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
from actiDep.utils.fod import get_tissue_responses, get_msmt_csd, get_peaks
import tempfile
import glob
import shutil

def process_msmt_csd(subject):
    """
    Process the MSMT-CSD pipeline on the given subject.
    """
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    if isinstance(subject, str):
        subject = Subject(subject)
    pipeline='msmt_csd'

    pipeline_list = ['init','response','fod','peaks']
    # pipeline_list = ['peaks']

    if 'init' in pipeline_list:
        create_pipeline_description(pipeline, layout=subject.layout, registrationMethod='antsRegistrationSyNQuick.sh', registrationParams='default', exactParams={'-d': '3'})

    
   
        
        
if __name__ == "__main__":
    config,tools = set_config()
    subject = Subject('03011')
 
    process_msmt_csd(subject)