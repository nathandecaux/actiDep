#!/bin/sh

cd groupe1

tract=/home/ndecaux/Code/actiDep/tests/599469/tracts/whole_brain.trk

ref=/home/ndecaux/Code/actiDep/tests/HCP_Sample/Diffusion_Data_Preprocessed/599469/Images/data_B0.nii.gz

transfo=./Transfos/DTI_599469.xml

# Apply transformation to the tract with animaApplyTransformSerie
mkdir -p ./Tracts
animaApplyTransformSerie -i $tract -g $ref -t $transfo -o ./Tracts/whole_brain_transformed.trk