import os
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
from actiDep.data.loader import Subject, Actidep, ActiDepFile
import tempfile
from subprocess import call

def average_with_ants(dataset, output_file=None):
    """
    Find all subject files with desc-fixels2peak and suffix=density,
    transform them to common space and create template using antsMultivariateTemplateConstruction2.sh
    """

    #Create a tempdir
    temp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory created at: {temp_dir.name}")
    if output_file is None:
        output_file = Path(temp_dir.name) / 'fiber_density_'
    files = dataset.get_global(pipeline='anima_preproc', metric='FA', extension='nii.gz')

    #Create a line separated string with all the files paths    
    file_list = '\n'.join([str(file.path) for file in files])
    print(file_list)
    file_list_path = Path(temp_dir.name) / 'file_list.txt'
    with open(file_list_path, 'w') as f:
        f.write(file_list)
    
    #Run antsMultivariateTemplateConstruction2.sh
    cmd = ['antsMultivariateTemplateConstruction2.sh',
           '-d', '3',
           '-o', str(output_file),
           '-c', '2',
           '-n', '0',
           '-j', '40',
            str(file_list_path)]
    
    print(f"Running command: {' '.join(cmd)}")
    call(cmd)

    print(f"Template created at: {output_file}")
    

# Example usage
if __name__ == "__main__":
    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    ds = Actidep(dataset_path)
    average_with_ants(ds)