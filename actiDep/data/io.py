from bids import BIDSLayout
import bids
from os.path import join as opj
import os
import SimpleITK as sitk
import shutil
import pathlib
from ..utils.tools import del_key, upt_dict,create_pipeline_description


def parse_filename(filename):
    """
    Parse the filename to extract the entities
    """
    entities = {}
    parts = filename.split('_')
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    return entities


def convertNRRDToNifti(nrrd_path, nifti_path):
    """Convertit un fichier NRRD en NIfTI."""
    # Lire l'image NRRD
    itk_image = sitk.ReadImage(nrrd_path)

    # Écrire l'image NIfTI
    sitk.WriteImage(itk_image, nifti_path)


def copy2nii(source, dest):
    """Copie un fichier, en convertissant en nifti si nécessaire."""
    print(f"Copying {source} to {dest}")
    pathlib.Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)

    if source.endswith(".nrrd") and dest.endswith(".nii.gz"):
        convertNRRDToNifti(source, dest)
    else:
        shutil.copy2(source, dest)
    return dest


def move2nii(source, dest):
    """Déplace un fichier, en convertissant en nifti si nécessaire."""
    print(f"Moving {source} to {dest}")
    pathlib.Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)

    if source.endswith(".nrrd") and dest.endswith(".nii.gz"):
        convertNRRDToNifti(source, dest)
        os.remove(source)
    else:
        shutil.move(source, dest)
    return dest

def copy_list(dest, file_list):
    """
    Copy a list of files to a new location after checking they exist.
    Files could contains BIDSFile, ActiDepFile, Path or str objects.
    """
    for f in file_list:
        src_path = ""
        if isinstance(f, str):
            src_path = f
        elif isinstance(f, pathlib.Path):
            src_path = str(f)
        elif isinstance(f, bids.layout.BIDSFile):
            src_path = f.path
        elif isinstance(f, ActiDepFile):
            src_path = f.path
        else:
            # raise ValueError(f"Unknown type {type(f)}")
            print(f"Unknown type {type(f)}")
            continue
        
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue
        else:
            shutil.copy(src_path, dest)

def copy_from_dict(subject, file_dict, pipeline=None,dry_run=False):
    """
    Copy files from a dictionary to the BIDS dataset.
    eg file_dict : 
    {'/tmp/tmpkm3ebons/t1_pve_0.nii.gz': {'datatype': 'anat',
                                      'extension': '.nii.gz',
                                      'label': 'CSF',
                                      'space': 'B0',
                                      'subject': '03011',
                                      'suffix': 'propseg'}
    }
    """
    for src_file, entities in file_dict.items():
        dest_file = subject.build_path(
            original_name=os.path.basename(src_file), **entities, pipeline=pipeline)
        if not dry_run:
            copy2nii(src_file, dest_file)
        else:
            print(f"Copying {src_file} to {dest_file}")