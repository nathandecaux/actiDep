import os
import subprocess
from pprint import pprint
import SimpleITK as sitk
import nibabel as nib

#!/usr/bin/env python3

# Define paths

base_dir = "/home/ndecaux/Code/actiDep/tests/groupe1"
os.chdir("/home/ndecaux/Code/actiDep/tests/groupe1")

tract = "/home/ndecaux/Code/actiDep/tests/HCP_Sample/HCP105_Zenodo/599469/tracts/CST_left.trk"
ref = "/home/ndecaux/Code/actiDep/tests/HCP_Sample/Structural_Data_Preprocessed/599469/Images/T1w_acpc_dc_restore_brain.nii.gz"
transfo = "./Transfos/DTI_599469.xml"

#Read the xml file
import xml.etree.ElementTree as ET

def read_transformation_xml(xml_file):
    """Read and parse an XML transformation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    transformations = []
    for transform in root.findall('Transformation'):
        transform_type = transform.find('Type').text
        inversion = bool(int(transform.find('Inversion').text))
        path = transform.find('Path').text
        
        transformations.append({
            'type': transform_type,
            'inversion': inversion,
            'path': path
        })
    
    return transformations

# Read the transformation XML file
transformations = read_transformation_xml(transfo)

# Import necessary DIPY modules
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines

# Load the reference image
print("Loading reference image:", ref)
ref_sitk = sitk.ReadImage(ref)
ref_nib = nib.load(ref)

# Load the tractogram
print("Loading tractogram:", tract)
tractogram = load_trk(tract, ref, trk_header_check=False)
streamlines = tractogram.streamlines

# Process each transformation
transformed_streamlines = streamlines
transformed_image = ref_sitk
for transform in transformations:
    transform_path = os.path.join(base_dir, transform['path'])
    print(f"Processing transformation: {transform_path}")
    
    if transform['type'] == 'linear':
        print(f"Applying linear transformation from {transform_path}")
        
        # Parse the ITK transform file
        sitk_transform = sitk.ReadTransform(transform_path)
        transform_matrix = np.eye(4)
        matrix = np.array(sitk_transform.GetParameters()).reshape(3, 4)
        transform_matrix[:3, :4] = matrix
        transform_matrix[:3, 3] = np.array(sitk_transform.GetFixedParameters())

        
        print("Transform matrix:")
        print(transform_matrix)
        
      
        
        # Convert ITK parameters to 4x4 affine matrix
        
        
        # If inversion is needed, invert the transformation matrix
        if transform['inversion']:
            transform_matrix = np.linalg.inv(transform_matrix)
        
        # Apply the linear transformation to streamlines
        transformed_streamlines = transform_streamlines(transformed_streamlines, transform_matrix)
        
        # Apply transformation to image
        if transform['inversion']:
            # For image transformation, we need to revert the inversion
            sitk_transform = sitk_transform.GetInverse()
        
        # Resample the image
        transformed_image = sitk.Resample(transformed_image, transformed_image, sitk_transform,
                                         sitk.sitkLinear, 0.0, transformed_image.GetPixelID())
        
    elif transform['type'] == 'svf':
        print(f"Applying non-linear transformation from {transform_path}")
        
        # Load the displacement field
        disp_img = sitk.ReadImage(transform_path)
        
        # Convert SimpleITK image to numpy array for streamlines transformation
        disp_field_array = sitk.GetArrayFromImage(disp_img)
        
        # SimpleITK to nibabel orientation convention
        disp_field_array = np.transpose(disp_field_array, (2, 1, 0, 3))
        
        # Get the affine transform from the displacement field
        direction = np.array(disp_img.GetDirection()).reshape(3, 3)
        spacing = np.array(disp_img.GetSpacing())
        origin = np.array(disp_img.GetOrigin())
        
        # Build the affine matrix
        affine = np.eye(4)
        affine[:3, :3] = direction * spacing
        affine[:3, 3] = origin
        
        from dipy.align.imwarp import DiffeomorphicMap
        
        # Create a DiffeomorphicMap object
        domain_shape = disp_field_array.shape[:3]
        diff_map = DiffeomorphicMap(3, domain_shape, affine)
        
        # Set the displacement field
        if transform['inversion']:
            # If inversion is needed, need to use the inverse displacement field
            print("Using inverse displacement field")
            diff_map.backward = disp_field_array
        else:
            diff_map.forward = disp_field_array
        
        # Apply the non-linear transformation to streamlines
        transformed_streamlines = diff_map.transform_points(
            transformed_streamlines,
        )
        
        # Apply the non-linear transformation to the image using SimpleITK
        # Create a displacement field transform
        if transform['inversion']:
            # If we need to invert, use a different approach with SimpleITK
            # For SVF, we can use the inverse displacement field
            displacement_transform = sitk.DisplacementFieldTransform(sitk.Cast(disp_img, sitk.sitkVectorFloat64))
            displacement_transform = displacement_transform.GetInverse()
        else:
            displacement_transform = sitk.DisplacementFieldTransform(sitk.Cast(disp_img, sitk.sitkVectorFloat64))
        
        # Apply the transform to the image
        transformed_image = sitk.Resample(transformed_image, transformed_image, displacement_transform,
                                         sitk.sitkLinear, 0.0, transformed_image.GetPixelID())

# Save the transformed tractogram
output_tract = os.path.join(os.path.dirname(tract), "whole_brain_transformed.trk")
tractogram.streamlines = transformed_streamlines
save_trk(tractogram, output_tract)
print(f"Transformed tractogram saved to: {output_tract}")

# Save the transformed image
output_image_path = os.path.join(os.path.dirname(ref), "ref_image_transformed.nii.gz")
sitk.WriteImage(transformed_image, output_image_path)
print(f"Transformed reference image saved to: {output_image_path}")
