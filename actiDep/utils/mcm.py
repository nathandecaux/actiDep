from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, move2nii, parse_filename, ActiDepFile, copy2nii
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, CLIArg, run_anima_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import xml.etree.ElementTree as ET

def read_mcm_file(mcm_file):
    """
    Read the MCM file and return dictionary containing model structure.
    
    Parameters
    ----------
    mcm_file : str
        Path to .mcm XML file
        
    Returns
    -------
    dict
        Dictionary containing weights file and compartment information
    """
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    
    model = {
        'weights': root.find('Weights').text,
        'compartments': []
    }
    
    for comp in root.findall('Compartment'):
        model['compartments'].append({
            'type': comp.find('Type').text,
            'filename': comp.find('FileName').text
        })
        
    return model


def mcm_estimator(dwi, bval, bvec, mask, n_comparts, **kwargs):
    """
    Calls the animaMCMEstimator to estimate the MCM coefficients from the DWI data.

    Parameters
    ----------
    dwi : ActiDepFile
        The DWI data to estimate the responses from
    bval : ActiDepFile
        The bval file associated with the DWI data
    bvec : ActiDepFile
        The bvec file associated with the DWI data
    mask : ActiDepFile
        Brain mask to use for the estimation
    n_comparts : int
        Number of anisotropic compartments to estimate
    kwargs : dict  
        Additional arguments to pass to the script
    """
    # Préparer les entrées
    inputs = {
        "dwi": dwi,
        "bval": bval,
        "bvec": bvec,
        "mask": mask
    }
    
    # Construire les arguments de commande avec des références symboliques (copie les fichiers dans un dossier temporaire)
    command_args = [
        "-b", "$bval",
        "-g", "$bvec",
        "-i", "$dwi",
        "-m", "$mask",
        "-o", "mcm.nii.gz",
        "-n", str(n_comparts)
    ]
    
    # Définir les sorties attendues (pattern de base, sera complété après exécution)
    output_pattern = {
        'mcm.nii.mcm': {"model": "MCM", "extension": ".mcm"}
    }

    # Exécuter la commande
    result = run_anima_command(
        "animaMCMEstimator",
        inputs,
        output_pattern,
        dwi,
        command_args=command_args,
        **kwargs
    )
    
    # Chemin vers le fichier MCM généré
    mcm_file = next(path for path in result.keys() if path.endswith('.mcm'))
    
    # Lire le modèle MCM pour identifier les compartiments
    mcm_model = read_mcm_file(mcm_file)
    tmp_folder = os.path.dirname(mcm_file)
    
    # Ajouter les compartiments au dictionnaire de résultats
    base_entities = dwi.get_entities() if isinstance(dwi, ActiDepFile) else dwi.copy()
    base_entities = upt_dict(base_entities, model='MCM', extension='nii.gz')
    
    # Ajouter les fichiers de compartiments
    for comp in mcm_model['compartments']:
        comp_path = opj('mcm.nii', comp['filename'])
        comp_num = comp['filename'].split('_')[-1].split('.')[0]  # Get number from filename
        result[opj(tmp_folder, comp_path)] = upt_dict(
            base_entities.copy(), 
            compartment=comp_num, 
            extension='.nii.gz',
            desc=comp['type'].lower()
        )
    
    # Ajouter le fichier de poids
    result[opj(tmp_folder, 'mcm.nii/mcm.nii_weights.nrrd')] = upt_dict(
        base_entities.copy(), 
        extension='.nii.gz',
        model='MCM',
        desc='weights'
    )
    
    return result


if __name__ == "__main__":
    config, tools = set_config()
    subject = Subject('03011')
    dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc', extension='nii.gz')
    bval = subject.get_unique(extension='bval')
    bvec = subject.get_unique(extension='bvec', desc='preproc')
    mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')
    
    res_dict = mcm_estimator(
        dwi, bval, bvec, mask, 3,
        R=True, F=True, S=True, c=2, 
        ml_mode=CLIArg('ml-mode', 2),
        opt=CLIArg('optimizer', 'levenberg')
    )
    
    pprint(res_dict)