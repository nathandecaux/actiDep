#!/bin/sh

BASE_DIR="/data/asavalle/P007033/NIFTI"

dwi="DICOM_TENSEUR_64DIR_AP_20230428000000_8_preprocessed.nii.gz"
bvals="DICOM_TENSEUR_64DIR_AP_20230428000000_8.bval"
bvecsAnima="DICOM_TENSEUR_64DIR_AP_20230428000000_8_preprocessed.bvec"
bvecs="DICOM_TENSEUR_64DIR_AP_20230428000000_8.bvec"
b0="DICOM_TENSEUR_64DIR_AP_20230428000000_8_Tensors_B0.nii.gz"
brain_mask="DICOM_TENSEUR_64DIR_AP_20230428000000_8_brainMask.nii.gz"
t1="DICOM_T1_MPRAGE_20230428000000_14_masked.nii.gz"
output_dir="out_nathan_avec_bvec_base"

# #For all files, provide the full path or the path relative to BASE_DIR
# dwi="$BASE_DIR/$dwi"
# bvals="$BASE_DIR/$bvals"
# bvecsAnima="$BASE_DIR/$bvecsAnima"
# bvecs="$BASE_DIR/$bvecs"
# b0="$BASE_DIR/$b0"
# t1="$BASE_DIR/$t1"
# output_dir="$BASE_DIR/$output_dir"

# BVEC="/data/asavalle/tractseg_comascore_P0701/P007001_DWI.bvec"
# DWI="/data/asavalle/tractseg_comascore_P0701/P007001_DWI_masked.nii.gz"
# T1="/data/asavalle/tractseg_comascore_P0701/FA_MNI_template.nii.gz"
# BVAL="/data/asavalle/tractseg_comascore_P0701/P007001_DWI.bval"
# BRAIN_MASK="/home/ndecaux/Code/Data/brain_mask_resliced.nii.gz"
# TEMPLATE="/data/HCP_Data/Structural_Data_Preprocessed/100206/Images/T1w_acpc_dc_restore.nii.gz"
# OUTDIR="test_tractseg"

BVEC="${BASE_DIR}/${bvecs}"
DWI="${BASE_DIR}/${dwi}"
T1="${BASE_DIR}/${t1}"
BVAL="${BASE_DIR}/${bvals}"
BRAIN_MASK="${BASE_DIR}/${brain_mask}"
TEMPLATE="/home/ndecaux/Data/test_coma/MNI_FA_template.nii.gz"
OUTDIR="/data/ndecaux/${output_dir}"

#--moving_source fa --invert_bvec_y_input --no_skullstrip --moving_source fa --invert_bvec_y_input
CMD="./tractseg.py --dwi ${DWI} --register_on fa --bvals ${BVAL} --bvecs ${BVEC} --atlas ${TEMPLATE} --out_dir ${OUTDIR} --brain_mask ${BRAIN_MASK}"
echo $CMD
$CMD
