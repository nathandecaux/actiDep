import os 
from pprint import pprint
from os.path import join as opj
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import xml.etree.ElementTree as ET
import multiprocessing
from functools import partial
import traceback
import tempfile
import dipy


base_dir = "/local/tdurante/atlas/groupe1/"
sub_list = "/local/tdurante/atlas/groupe1/sujets.txt"
bundle_list = "/local/tdurante/atlas/groupe1/bundles.txt"
# bundle_src="/data/HCP_Data/HCP105_Zenodo/"
bundle_src="/data/HCP_Data/Diffusion_Data_Preprocessed/"
# bundle_src='/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/'
anat_src="/data/HCP_Data/Structural_Data_Preprocessed/"
output_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/"
#Bundle path example /data/HCP_Data/HCP105_Zenodo/599469/tracts/AF_left.vtp


with open(sub_list, 'r') as f:
    subjects = f.read().splitlines()

with open(bundle_list, 'r') as f:
    bundles = f.read().splitlines()

subjects = {k+1:v for k,v in enumerate(subjects)}

def create_fake_rigid_transform():
    """
    Create a fake rigid transformation matrix for testing purposes.
    """
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = 10  # Translate x by 10
    transform_matrix[1, 3] = -11  # Translate y by -5
    transform_matrix[2, 3] = 12  # Translate z by 2
    return transform_matrix

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

def convert_transform_matrix_to_nibabel(transform_matrix):
    """
    Convert a transformation matrix from SimpleITK to nibabel format.
    SimpleITK uses LPS (Left-Posterior-Superior) coordinate system,
    while nibabel uses RAS (Right-Anterior-Superior).
    This function handles the conversion between these coordinate systems.
    """
    # Create the coordinate system conversion matrix (LPS to RAS)
    lps_to_ras = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the matrix by applying the coordinate system transformation
    nib_matrix = lps_to_ras @ transform_matrix @ np.linalg.inv(lps_to_ras)
    
    return nib_matrix

def apply_transform_to_tract(tract_path, ref_path, transform_xml_path, output_dir=None, transform_ref=False):
    """
    Apply transformations defined in XML file to a tractography file and optionally to the reference image.
    
    Parameters:
    -----------
    tract_path : str
        Path to the tract file (.vtp)
    ref_path : str
        Path to the reference image (.nii.gz)
    transform_xml_path : str
        Path to the transformation XML file
    output_dir : str, optional
        Directory to save output files. If None, uses the same directory as input files.
    transform_ref : bool, optional
        Whether to transform the reference image as well.
        
    Returns:
    --------
    dict
        Dictionary with paths to transformed files
    """
    base_dir = os.path.dirname(os.path.dirname(transform_xml_path))
    os.chdir(base_dir)
    tract_name = os.path.basename(tract_path).replace('.vtp', '')
    
    # Read the transformation XML file
    # transformations = read_transformation_xml(transform_xml_path)
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(tract_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the reference image if needed
    # moved_ref_path=f"{output_dir}/transformed_ref.nii.gz" if transform_ref else None
    
    if transform_ref:
        # Use animaApplyTransformSerie
        print(f"Transforming reference image: {ref_path}")
        trans_cmd = f"animaApplyTransformSerie -i {ref_path} -g {ref_path} -o {output_dir}/transformed_anat.nii.gz -t {transform_xml_path}"
        os.system(trans_cmd)

    return True
    # #Use animaFibersApplyTransformSerie
    # trans_cmd = f"animaFibersApplyTransformSerie -i {tract_path} -o {output_dir}/transformed_{tract_name}.vtk -t {transform_xml_path}"

    # print(trans_cmd)

    # os.system(trans_cmd)

    # print(f"Transformation completed, output saved to {output_dir}/transformed_{tract_name}.vtk")

    # print(f"Converting transformed VTP to TRK")

    # convert_cmd = f"dipy_convert_tractogram {output_dir}/transformed_{tract_name}.vtk --out_tractogram {output_dir}/transformed_{tract_name}.trk --reference {ref_path} --force"

    # os.system(convert_cmd)

    # return {
    #     'tract': f"{output_dir}/transformed_{tract_name}",
    # }

# Fonction auxiliaire pour traiter un faisceau (à ajouter avant la boucle principale)
def process_bundle(bundle, bundle_src, sub_name, ref_path, transfo, sub_out_dir):
    bundle_path = opj(bundle_src, sub_name, "Tracks", f"{bundle}.vtp")
    excepted_out_bundle = opj(sub_out_dir, f"transformed_{bundle}.vtp")
    
    if os.path.exists(excepted_out_bundle):
        print(f"Bundle {bundle} already processed, skipping")
        return
    
    try:
        print(f"Processing bundle: {bundle}")
        res = apply_transform_to_tract(bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=False)
        print(f"Bundle {bundle} processed successfully")
    except Exception as e:
        error_file = opj(sub_out_dir, f"error_{bundle}.txt")
        with open(error_file, 'w') as f:
            f.write(str(e))
        print(f"Error processing bundle {bundle}, check {error_file}")
        # Print the full error message to console
        print(f"Detailed error: {str(e)}")
        print(traceback.format_exc())


# Only process the subject 599671
#Get hostname
hostname = os.uname()[1]
print(f"Hostname: {hostname}")
if hostname == "calcarine":
    #Get the first half of the subjects
    subjects = list(subjects.items())[:len(subjects)//2]
else:
    #Get the second half of the subjects
    subjects = list(subjects.items())[len(subjects)//2:]

# Convert to dictionary
subjects = {k:v for k,v in subjects}
# subjects = {k:v for k,v in subjects if v == '871964'}

for sub_id, sub_name in subjects.items():
    transfo = opj(base_dir, f'Transfos/DTI_{sub_name}.xml')
    ref_path = opj(anat_src, sub_name, "Images", "T1w_acpc_dc_restore_brain.nii.gz")
    sub_out_dir = opj(output_dir, sub_name)
    os.makedirs(sub_out_dir, exist_ok=True)
    print(f"Processing subject {sub_id}: {sub_name}")
    
    # Traiter le premier faisceau séparément pour transformer la référence
    first_bundle = bundles[0]
    # first_bundle= 'CC'
    first_bundle_path = opj(bundle_src, sub_name, "Tracks", f"{first_bundle}.vtp")
    excepted_out_bundle = opj(sub_out_dir, f"transformed_{first_bundle}.vtp")
    
    if not os.path.exists(excepted_out_bundle):
        try:
            print(f"Processing first bundle with reference transformation: {first_bundle}")
            res = apply_transform_to_tract(first_bundle_path, ref_path, transfo, output_dir=sub_out_dir, transform_ref=True)
            print("Results:")
            pprint(res)
        except Exception as e:
            error_file = opj(sub_out_dir, f"error_{first_bundle}.txt")
            with open(error_file, 'w') as f:
                f.write(str(e))
            print(f"Error processing bundle {first_bundle}, check {error_file}")
            print(f"Detailed error: {str(e)}")
    
    # # Traiter les autres faisceaux en parallèle sans transformer la référence
    # remaining_bundles = bundles[1:] if bundles else []
    # if remaining_bundles:
    #     # Créer une fonction partielle avec les arguments constants
    #     process_bundle_partial = partial(
    #         process_bundle, 
    #         bundle_src=bundle_src,
    #         sub_name=sub_name,
    #         ref_path=ref_path,
    #         transfo=transfo,
    #         sub_out_dir=sub_out_dir
    #     )
        
    #     # Utiliser un pool de processus pour traiter les faisceaux en parallèle
    #     with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(remaining_bundles))) as pool:
    #         pool.map(process_bundle_partial, remaining_bundles)
    
    # print("\n\n")


