from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, move2nii, parse_filename, ActiDepFile, copy2nii,copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import dipy
from time import process_time


# command = [
#     "tckgen",
#     "-algorithm", "iFOD2",
#     "-seed_image", seeds,
#     odf_mrtrix,
#     output,
#     "-force"
# ]

def get_ifod2(odf, seeds, **kwargs):

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


if __name__ == "__main__":
    sub = Subject("03011")
    odf = sub.get_unique(suffix='fod',  desc='preproc', label='WM')
    seeds = sub.get_unique(suffix='mask', label='WM', space='B0')

    output_dict=get_ifod2(odf, seeds)
    # pprint(output_dict)
    copy_from_dict(sub, output_dict,pipeline='msmt_csd')