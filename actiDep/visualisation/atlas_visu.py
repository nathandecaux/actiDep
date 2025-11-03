import os 
from pprint import pprint
from os.path import join as opj
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import read_mapping
import multiprocessing
from functools import partial


sub_list = "/local/tdurante/atlas/groupe1/sujets.txt"
anat_src="/data/HCP_Data/Structural_Data_Preprocessed/"
output_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/"

