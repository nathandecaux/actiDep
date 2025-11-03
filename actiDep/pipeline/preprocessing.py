import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.utils.tools import upt_dict
from actiDep.data.loader import Subject
from actiDep.data.io import copy_from_dict
import tempfile
import glob
import shutil

pipeline = 'anima_preproc'

def flip_bvec_y(bvec_path, output_path):
    """
    Flip bvec in y direction by multiplying y component by -1
    """
    bvec_data = np.loadtxt(bvec_path)
    if bvec_data.shape[0] == 3:  # Standard format: 3 rows (x,y,z), N columns
        bvec_data[1, :] *= -1  # Flip y component
    elif bvec_data.shape[1] == 3:  # Alternative format: N rows, 3 columns (x,y,z)
        bvec_data[:, 1] *= -1  # Flip y component
    np.savetxt(output_path, bvec_data, fmt='%.6f')

def animaPreprocessing(subject, with_reversed_b0 = True, db_root='/home/ndecaux/Code/Data/dysdiago'):
    """
    Calls the Anima diffusion preprocessing script on the given subject.
    """
    
    # Generate temporary folder in /local/ndecaux/tmp/
    import time
    temp_dir = f"/local/ndecaux/tmp/anima_preproc_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)
    
    if isinstance(subject, str):
        subject = Subject(subject, db_root)

    # Preprocess diffusion data
    print("Preprocess diffusion data")
    config,tools = set_config()

    dwi = subject.get(suffix='dwi',extension='nii.gz',scope='raw')[0]

    t1 = subject.get(suffix='T1w',scope='raw',extension='nii.gz')[0]
    b0_reversed= subject.get_unique(desc='b0reversed',scope='raw',extension='nii.gz')
    dicom_folder = subject.dicom_folder
    
    # Copy files to temp directory and flip bvec
    temp_dwi = os.path.join(temp_dir, os.path.basename(dwi.path))

    temp_t1 = os.path.join(temp_dir, os.path.basename(t1.path))
    
    shutil.copy2(dwi.path, temp_dwi)
    shutil.copy2(t1.path, temp_t1)


    bval = subject.get(suffix='dwi',scope='raw',extension='bval')[0]
    temp_bval = os.path.join(temp_dir, os.path.basename(bval.path))
    shutil.copy2(bval.path, temp_bval)


    bvec = subject.get(suffix='dwi',scope='raw',extension='bvec')[0]
    temp_bvec = os.path.join(temp_dir, os.path.basename(bvec.path))

    # Flip bvec in y direction
    flip_bvec_y(bvec.path, temp_bvec)
    
    if with_reversed_b0:
        temp_b0_reversed = os.path.join(temp_dir, os.path.basename(b0_reversed.path))
        shutil.copy2(b0_reversed.path, temp_b0_reversed)
    
    # Change to temp directory for processing
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    #Get directory and prefix of dwi file
    dwiPrefix = os.path.basename(dwi.path).split('.')[0] 
    
    #Get files with full path
    dicom_files = [f for f in glob.glob(os.path.join(dicom_folder, "**", "*"), recursive=True) if os.path.isfile(f)]
    print(dicom_files[:10])

    preprocCommand = ["python3", tools['animaDiffusionImagePreprocessing'], "-t", temp_t1, "-i", temp_dwi,'--temp-folder',temp_dir]

    preprocCommand +=  ["-D"] + dicom_files
    preprocCommand = preprocCommand + ["-b", temp_bval]
    # preprocCommand = preprocCommand + ["-g", temp_bvec]
    # preprocCommand = preprocCommand + ["--no-disto-correction"]
    # preprocCommand = preprocCommand + ["--no-eddy-correction","--no-denoising"]
    if with_reversed_b0:
        preprocCommand = preprocCommand + ["-r", temp_b0_reversed]

    print(preprocCommand)
    call(preprocCommand)
    
    print("Preprocess data finished")
    # os.chdir(original_dir)
    
    entities = {'suffix':'dwi','pipeline':pipeline}
    # pipeline = 'anima_preproc'
    
    tensors = dwiPrefix + "_Tensors.nrrd"
    preprocessed_dwi = dwiPrefix + "_preprocessed.nrrd"
    preprocessed_bvec = dwiPrefix + "_preprocessed.bvec"
    brain_mask = dwiPrefix + "_brainMask.nrrd"


    entities=dwi.get_full_entities()

    res_dict= {
        tensors: upt_dict(entities, {'datatype': 'dwi', 'desc': 'tensors', 'model': 'DTI'}),
        preprocessed_dwi: upt_dict(entities, {'datatype': 'dwi', 'desc': 'preproc'}),
        preprocessed_bvec: upt_dict(entities, {'datatype': 'dwi', 'desc': 'preproc', 'extension': 'bvec'}),
        brain_mask: upt_dict(entities, {'label': 'brain', 'suffix': 'mask','space':'B0','datatype': 'anat', 'pipeline': pipeline})
    }
    

    
    print(res_dict)

    copy_from_dict(subject,res_dict,pipeline=pipeline)
    # move2nii(tensors, tensors_target)
    # move2nii(preprocessed_dwi, prepocessed_dwi_target)
    # shutil.move(preprocessed_bvec, preprocessed_bvec_target)
    # move2nii(brain_mask, brain_mask_target)
    
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_B0.nrrd"))
    # os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_NoiseVariance.nrrd"))

    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
def compute_fa(subject, db_root='/home/ndecaux/Code/Data/dysdiago'):
    """
    Computes FA from the preprocessed diffusion data.
    """
    config,tools = set_config()
    
    subject = Subject(subject, db_root)
    
    tensors = subject.get_unique(suffix='dwi',pipeline=pipeline,model='DTI',metric=None)
    print(tensors.path)
    #create temporary directory
    temp_dir = tempfile.mkdtemp()

    fa_command = f'{tools["animaDTIScalarMaps"]} -i {tensors.path} -f {temp_dir}/fa.nii.gz'
    
    call(fa_command, shell=True)

    res_dict = {
        f'{temp_dir}/fa.nii.gz': upt_dict(tensors.get_full_entities(), metric='FA')
    }
    print(tensors.get_full_entities())
    copy_from_dict(subject, res_dict, pipeline=pipeline)

def process_all_FA(db_root='/home/ndecaux/Code/Data/dysdiago'):
    """
    Process all subjects in the database to compute FA.
    """
    config,tools = set_config()
    
    from actiDep.data.loader import Actidep
    ds = Actidep(db_root)
    
    subjects = ds.get_subjects()
    
    for subject in subjects:
        try:
            print(f"Processing subject {subject} for FA computation.")
            compute_fa(subject, db_root=db_root)
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
        

if __name__ == '__main__':
    config,tools = set_config()
    # animaPreprocessing('01',db_root="/home/ndecaux/NAS_EMPENN/share/projects/actidep/IRM_Cerveau_MOI/bids")
    # animaPreprocessing('02')
    # animaPreprocessing('03026',db_root="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids")
    # compute_fa('01',db_root="/home/ndecaux/NAS_EMPENN/share/projects/actidep/IRM_Cerveau_MOI/bids")
    # compute_fa('02')

    amynet='/home/ndecaux/NAS_EMPENN/share/projects/amynet/bids'
    process_all_FA(db_root=amynet)