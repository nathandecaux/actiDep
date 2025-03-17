import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, move2nii
import tempfile
import glob
import shutil

def animaPreprocessing(subject, with_reversed_b0 = True):
    """
    Calls the Anima diffusion preprocessing script on the given subject.
    """
    
    #Generate temporary folder to work in
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    
    if isinstance(subject, str):
        subject = Subject(subject)

    # Preprocess diffusion data
    print("Preprocess diffusion data")
    config,tools = set_config()

    dwi = subject.get(suffix='dwi',scope='raw',extension='nii.gz')[0]
    bval = subject.get(suffix='dwi',scope='raw',extension='bval')[0]
    bvec = subject.get(suffix='dwi',scope='raw',extension='bvec')[0]
    t1 = subject.get(suffix='T1w',scope='raw',extension='nii.gz')[0]
    dicom_folder = subject.dicom_folder
    
    #Get directory and prefix of dwi file
    #Directory
    dwiPrefixBase = os.path.dirname(dwi.path)
    #Prefix
    dwiPrefix = os.path.basename(dwi.path).split('.')[0] 
    
    
    #Get files with full path
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))

    preprocCommand = ["python3", tools['animaDiffusionImagePreprocessing'], "-b", bval.path, "-t", t1.path, "-i", dwi.path, "-D"] + dicom_files
    # print(preprocCommand)

    if with_reversed_b0:
        preprocCommand = preprocCommand + ["-r", os.path.join(subject.db_root, subject.bids_id, "dwi", f"{subject.bids_id}_dwi_reversed_b0.nii.gz")]

    # preprocCommand = preprocCommand + ["-g", os.path.join(subject.db_root, subject.bids_id, "dwi", f"{subject.bids_id}_dwi.bvec")]

    # call(preprocCommand)
    
    print("Preprocess data finished")
    os.chdir(dwiPrefixBase)

    pipeline = 'anima_preproc_test'
    
    entities = {'suffix':'dwi','pipeline':pipeline}
    # pipeline = 'anima_preproc'
    
    tensors = dwiPrefix + "_Tensors.nrrd"
    preprocessed_dwi = dwiPrefix + "_preprocessed.nrrd"
    preprocessed_bvec = dwiPrefix + "_preprocessed.bvec"
    brain_mask = dwiPrefix + "_brainMask.nrrd"
    
    tensors_target = subject.build_path(original_name=tensors,model='DTI',**entities)
    prepocessed_dwi_target = subject.build_path(**entities,desc='preproc')
    preprocessed_bvec_target = subject.build_path(**entities,desc='preproc',extension='bvec')
    brain_mask_target = subject.build_path(label='brain',suffix='mask',datatype='dwi',pipeline=pipeline)
    
    print(tensors_target)
    print(prepocessed_dwi_target)
    print(preprocessed_bvec_target)
    print(brain_mask_target)
    # move2nii(tensors, tensors_target)
    # move2nii(preprocessed_dwi, prepocessed_dwi_target)
    # shutil.move(preprocessed_bvec, preprocessed_bvec_target)
    # move2nii(brain_mask, brain_mask_target)
    
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_B0.nrrd"))
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_NoiseVariance.nrrd"))

    
    
if __name__ == '__main__':
    config,tools = set_config()
    animaPreprocessing('03011')