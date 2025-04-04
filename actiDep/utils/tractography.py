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
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from time import process_time
import vtk


def rotation(in_file, out_file, angle=180, x=0, y=0, z=1):
    '''Rotate a vtp or vtk file by ${angle}° along x, y or z axis.'''
    if in_file[-3:] == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
    elif in_file[-3:] == 'vtk':
        reader = vtk.vtkPolyDataReader()
        writer = vtk.vtkPolyDataWriter()
    else:
        print("Unrecognized input file. Must be vtp or vtk.")
        return
    reader.SetFileName(in_file)
    reader.Update()
    translation = vtk.vtkTransform()
    translation.RotateWXYZ(angle, x, y, z)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()
    writer.SetFileName(out_file)
    writer.SetInputConnection(transformFilter.GetOutputPort())
    # the constant 42 is defined in IO/Legacy/vtkDataWriter.h (c++ code), it corresponds to VTK_LEGACY_READER_VERSION_4_2
    writer.SetFileVersion(42)
    writer.Update()
    writer.Write()


# command = [
#     "tckgen",
#     "-algorithm", "iFOD2",
#     "-seed_image", seeds,
#     odf_mrtrix,
#     output,
#     "-force"
# ]

def generate_ifod2_tracto(odf, seeds, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    command_args = [
        "$odf",
        "-algorithm", "iFOD2",
        "-seed_image", "$seeds",
        "-force",
        "-debug",
        "tracto.tck"
    ]

    output_patterns = {
        "tracto.tck": {
            "suffix": "tracto",
            "datatype": "dwi",
            "extension": "tck"
        }
    }
    return run_mrtrix_command('tckgen', inputs, output_patterns, entities_template=odf.get_entities(), command_args=command_args, **kwargs)


def generate_trekker_tracto(odf, seeds, n_seeds=1000, **kwargs):

    inputs = {
        "odf": odf,
        "seeds": seeds
    }

    # odf_to_lps = run_cli_command('convert_fod', {'odf': odf}, {'odf_lps.nii.gz': odf.get_entities()}, command_args=[
    #                              '-i', odf.path, '-o', 'odf_lps.nii.gz', '-c', 'MRTRIX2ANIMA'])
    # #Get first key of odf_to_lps
    # odf_to_lps = list(odf_to_lps.keys())[0]

    command_args = [
        "track",
        odf.path,
        "--seed", seeds if isinstance(seeds, str) else seeds.path,
        "--seed_count", str(n_seeds),
        "-o", "tracto.vtk",
        "--force"
    ]

    output_patterns = {
        "tracto.vtk": {
            "suffix": "tracto",
            "datatype": "tracto",
            "extension": "vtk"
        }
    }
    result_dict = run_cli_command('trekker_linux', inputs, output_patterns, entities_template=odf.get_entities(
    ), command_args=command_args, **kwargs, use_sym_link=True)
    tract_path, tract_entities = list(result_dict.items())[0]

    # Rotate the VTK file
    rotated_path = tract_path.replace('.vtk', '_rotated.vtk')
    rotation(tract_path, rotated_path, angle=180, x=0, y=0, z=1)

    # # Convert VTK to TCK
    # call(["trekker_linux", "convert", tract_path, tract_path.replace('.vtk','.tck'), '--force'])

    result_dict[tract_path.replace('.vtk', '_rotated.vtk')] = upt_dict(tract_entities, orient='LPS')
    
    return result_dict


if __name__ == "__main__":
    sub = Subject("03011")
    odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    seeds = sub.get_unique(suffix='mask', label='WM', space='B0')

    output_dict = generate_ifod2_tracto(odf, seeds)
    # pprint(output_dict)
    # copy_from_dict(sub, output_dict,pipeline='msmt_csd')
