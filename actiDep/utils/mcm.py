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
from actiDep.data.mcmfile import MCMFile, read_mcm_file
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, CLIArg, run_anima_command,run_cli_command
from actiDep.utils.converters import call_dipy_converter
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import xml.etree.ElementTree as ET
import zipfile


def mcm_estimator(dwi, bval, bvec, mask, compart_map=None, n_comparts=None, **kwargs):
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
        "-o", "mcm"
        # "-n", str(n_comparts)
    ]

    if compart_map is not None:
        inputs['compart_map'] = compart_map
        command_args += ["-I", '$compart_map']
    elif n_comparts is not None:
        command_args += ["-n", str(n_comparts)]
    else:
        raise ValueError("Either compart_map or n_comparts must be provided.")
    
    # Définir les sorties attendues (pattern de base, sera complété après exécution)
    output_pattern = {
        'mcm.mcm': {"model": "MCM", "extension": ".mcm"}
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

    mcm_obj = MCMFile(mcm_file)
    mcm_obj.write(mcm_file.replace('.mcm', '.mcmx'))
    base_entities = dwi.get_entities() if isinstance(dwi, ActiDepFile) else dwi.copy()
    result={
        mcm_file.replace('.mcm', '.mcmx'): upt_dict(
            base_entities,
            model='MCM',
            extension='.mcmx'
        )
    }

    
    # # Lire le modèle MCM pour identifier les compartiments
    # mcm_model = read_mcm_file(mcm_file)
    # tmp_folder = os.path.dirname(mcm_file)
    
    # # Ajouter les compartiments au dictionnaire de résultats
    # base_entities = dwi.get_entities() if isinstance(dwi, ActiDepFile) else dwi.copy()
    # base_entities = upt_dict(base_entities, model='MCM', extension='nrrd')
    
    # # Ajouter les fichiers de compartiments
    # for comp in mcm_model['compartments']:
    #     comp_path = opj('mcm', comp['filename'])
    #     comp_num = comp['filename'].split('_')[-1].split('.')[0]  # Get number from filename
    #     result[opj(tmp_folder, comp_path)] = upt_dict(
    #         base_entities.copy(), 
    #         compartment=comp_num, 
    #         extension='.nrrd',
    #         type=comp['type'].lower()
    #     )
    
    # # Ajouter le fichier de poids
    # result[opj(tmp_folder, 'mcm/mcm_weights.nrrd')] = upt_dict(
    #     base_entities.copy(), 
    #     extension='.nrrd',
    #     model='MCM',
    #     type='weights'
    # )
    
    return result

def update_mcm_info_file(copy_dict):
    """
    Update the MCM info file with the new output directory.
    
    Parameters
    ----------
    copy_dict : dict
        Dictionary containing the mapping of source files to destination files
    """
    print(copy_dict)
    mcm_file = [v for k,v in copy_dict.items() if k.endswith('.mcm')][0]
    
    copy_dict = {os.path.basename(k): v for k, v in copy_dict.items() if not k.endswith('.mcm')}
    if mcm_file is None:
        raise ValueError("No MCM file found in the copy dictionary.")
    
    
    # Update in the XML file using copy_dict (replace copy_dict.keys() strings in XML by copy_dict.values())
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    for elem in root.iter():
        if (elem.tag == 'FileName' or elem.tag=='Weights') and elem.text in copy_dict:
            elem.text = os.path.basename(copy_dict[elem.text])

    # Write back to the file
    tree.write(mcm_file)
    return True

def add_mcm_to_tracts(tracts, mcm_file, reference=None,**kwargs):
    """
    Add MCM information to tracts.
    
    Parameters
    ----------
    tracts : ActiDepFile
        The tracts to which MCM information will be added
    mcm_file : ActiDepFile
        The MCM file containing the compartment information (.mcmx)
    """
    print("Adding MCM to tracts...")
    inputs = {
        "tracts": tracts,
        "mcm_file": mcm_file
    }

    tracts_extension = os.path.basename(tracts.path).split('.')[-1]

    if tracts_extension != 'vtk':
        converted_path= tracts.path.replace('.'+tracts_extension, '.vtk')
        call_dipy_converter(
            tracts.path,
            reference=reference,
            output_path=converted_path
        )
    else:
        converted_path = tracts.path
    mcm_obj = MCMFile(mcm_file.path)
    
    command_args = [
        "-i", '$tracts',
        "-m", mcm_obj.mcmfile,
        "-o", "tracts_mcm.vtk"
    ]
    # Define the expected output

    output_pattern = {
        'tracts_mcm.vtk': {"model": "MCM"}
    }

    # Run the command
    result = run_cli_command(
        "animaAddMCMToTracts",
        inputs,
        output_pattern,
        tracts,
        command_args=command_args
    )

    pprint(result)
    return result

    

if __name__ == "__main__":
    config, tools = set_config()
    # subject = Subject('03011')
    # dwi = subject.get_unique(suffix='dwi', desc='preproc', pipeline='anima_preproc', extension='nii.gz')
    # bval = subject.get_unique(extension='bval')
    # bvec = subject.get_unique(extension='bvec', desc='preproc')
    # mask = subject.get_unique(suffix='mask', label='brain', datatype='dwi')
    
    # res_dict = mcm_estimator(
    #     dwi, bval, bvec, mask, 3,
    #     R=True, F=True, S=True, c=2, 
    #     ml_mode=CLIArg('ml-mode', 2),
    #     opt=CLIArg('optimizer', 'levenberg')
    # )
    # pprint(res_dict)

    mcmfile = MCMFile('/tmp/tmpq4y0e7ei/test/mcm.mcm')
    mcmfile.write('/tmp/tmpq4y0e7ei/test.mcmx')
    mcmfile = MCMFile('/tmp/tmpq4y0e7ei/test.mcmx')
    print(mcmfile.compartments)