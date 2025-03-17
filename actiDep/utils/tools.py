import os
import json
from os.path import join as opj
from pathlib import Path
import pathlib
from os.path import join as opj
from bids.layout import BIDSFile
import shutil

def del_key(dct, key):
    d = dct.copy()
    del d[key]
    return d

def upt_dict(dct, new_items=None, **kwargs):
    """
    Update a dictionary with new items and/or keyword arguments. Not sure if it's very compliant with Python 
    conventions, but it works.

    Parameters
    ----------
    dct : Initial dictionary
    new_items : dict
        Dictionary of new items to add or update in the dictionary
    **kwargs : named arguments
        Named arguments to add or update in the dictionary

    """
    d = dct.copy()
    
    # Traiter le dictionnaire positionnel s'il existe
    if new_items and isinstance(new_items, dict):
        d.update(new_items)
        
    # Traiter les arguments nommés comme avant
    for key, value in kwargs.items():
        if isinstance(value, dict):
            d.update(value)
        else:
            if key != '':
                d[key] = value
            else:
                raise ValueError("Empty key. Please provide only named arguments.")
    return d

def create_pipeline_description(pipeline, layout,**kwargs):
    """
    Create the dataset_description.json file for the given subject and pipeline.
    """
    dataset_description = {
        "Name": f"actiDep {pipeline} pipeline",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "PipelineDescription": {
            "Name": pipeline,
            "Version": "0.1"
        }
    }
    dataset_description['PipelineDescription'].update(kwargs)

    #If pipeline folder does not exist, create it (pathlib)
    Path(opj(layout.root,'derivatives',pipeline)).mkdir(parents=True, exist_ok=True)
    
    with open(opj(layout.root,'derivatives',pipeline,'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)
    
def get_exact_file(list_of_files):
    """
    Return the file in the list if there is only one, else raise an error.
    """
    if len(list_of_files) != 1:
        raise ValueError(f"Found {len(list_of_files)} files")
    return list_of_files[0]

class CLIArg:
    def __init__(self, name, value=None, long_flag=True):
        self.name = name
        self.value = value
        self.long_flag = long_flag

        if long_flag:
            #Add the -- to the name if it is not already there
            self.name = self.name.lstrip('-')  # Remove any leading hyphens
            self.name = '--' + self.name
        else:
            #Add single - if not already there and remove any double --
            self.name = self.name.replace('--', '-')
            if not self.name.startswith('-'):
                self.name = '-'+self.name

    def get(self):
        if self.value:
            return [f"{self.name}", str(self.value)]
        else:
            return [f"{self.name}"]
    

def add_kwargs_to_cli(command,**kwargs):
    """
    Add the named arguments to the command line command.
    """
    if not kwargs:
        return command
    else:
        for key, value in kwargs.items():
            if isinstance(value,CLIArg):
                command=command+value.get()
            else:
                if isinstance(value,bool) and value:
                    command = command + [f"-{key.replace('-','')}"]
                else :
                    command = command + [f"-{key.replace('-','')}", str(value)]
        return command