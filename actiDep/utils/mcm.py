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
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, CLIArg, run_anima_command, run_cli_command
from actiDep.utils.converters import call_dipy_converter
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import ants
import xml.etree.ElementTree as ET
import zipfile
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata, interpn
import numpy as np
import nibabel as nib
import time


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
    result = {
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
    mcm_file = [v for k, v in copy_dict.items() if k.endswith('.mcm')][0]

    copy_dict = {
        os.path.basename(k): v
        for k, v in copy_dict.items() if not k.endswith('.mcm')
    }
    if mcm_file is None:
        raise ValueError("No MCM file found in the copy dictionary.")

    # Update in the XML file using copy_dict (replace copy_dict.keys() strings in XML by copy_dict.values())
    tree = ET.parse(mcm_file)
    root = tree.getroot()
    for elem in root.iter():
        if (elem.tag == 'FileName'
                or elem.tag == 'Weights') and elem.text in copy_dict:
            elem.text = os.path.basename(copy_dict[elem.text])

    # Write back to the file
    tree.write(mcm_file)
    return True


def convert_vtk_to_csv(vtk_file_path, output_base_path):
    """
    Convert a VTK PolyData file to multiple CSV files:
    - One CSV for fiber connectivity (FiberID, PointOrder, PointID).
    - One CSV per point data array (X, Y, Z, ArrayComponent1, ArrayComponent2, ...).

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK file.
    output_base_path : str
        Base path for the output CSV files (e.g., '/path/to/output/prefix').
        Array names or 'fibers' will be appended (e.g., prefix_ArrayName.csv).

    Returns
    -------
    dict
        A dictionary mapping descriptive keys (like 'fibers' or array names)
        to the paths of the generated CSV files.
        Returns an empty dictionary if conversion fails.
    """
    generated_csv_files = {}

    # Create a reader for VTK files (assuming legacy .vtk format)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    if reader.GetOutput() is None or reader.GetOutput().GetNumberOfPoints() == 0:
        print(
            f"Error: Could not read VTK file or file is empty: {vtk_file_path}")
        return generated_csv_files

    polydata = reader.GetOutput()

    # 1. Extract Point Data Arrays
    point_data = polydata.GetPointData()
    num_points = polydata.GetNumberOfPoints()

    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        if not array_name:
            array_name = f"UnnamedArray_{i}"

        num_components = array.GetNumberOfComponents()

        csv_file_path = f"{output_base_path}_{array_name}.csv"

        data_for_df = []
        header = ['X', 'Y', 'Z']
        if num_components == 1:
            header.append(array_name)
        else:
            for comp_idx in range(num_components):
                header.append(f"{array_name}_{comp_idx}")

        for pt_id in range(num_points):
            coords = polydata.GetPoint(pt_id)
            row = list(coords)
            value_tuple = array.GetTuple(pt_id)
            row.extend(list(value_tuple))
            data_for_df.append(row)

        df = pd.DataFrame(data_for_df, columns=header)
        df.to_csv(csv_file_path, index=False)
        generated_csv_files[array_name] = csv_file_path

    # 2. Extract Fiber Connectivity (Lines)
    lines = polydata.GetLines()
    if lines and lines.GetNumberOfCells() > 0:
        csv_fibers_path = f"{output_base_path}_fibers.csv"

        fiber_data_for_df = []
        header_fibers = ['FiberID', 'PointOrder', 'PointID']

        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        fiber_id_counter = 0
        while lines.GetNextCell(id_list):
            for point_order in range(id_list.GetNumberOfIds()):
                point_id = id_list.GetId(point_order)
                fiber_data_for_df.append(
                    [fiber_id_counter, point_order, point_id])
            fiber_id_counter += 1

        df_fibers = pd.DataFrame(fiber_data_for_df, columns=header_fibers)
        df_fibers.to_csv(csv_fibers_path, index=False)
        generated_csv_files['fibers'] = csv_fibers_path

    return generated_csv_files


def generate_scalar_map(vtk_file_path, reference_nifti_path, output_dir, output_file_prefix="scalar_map"):
    """
    Generates NIfTI scalar maps from each point data array in a VTK PolyData file,
    interpolating the values onto the grid of a reference NIfTI image.

    Parameters
    ----------
    vtk_file_path : str
        Path to the input VTK PolyData file (.vtk).
    reference_nifti_path : str
        Path to the reference NIfTI image file.
    output_dir : str
        Directory where the output NIfTI files will be saved.
    output_file_prefix : str
        Prefix for the output NIfTI file names. Array name will be appended.

    Returns
    -------
    list
        A list of paths to the generated NIfTI files.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_nifti_paths = []

    name_mapping = {
        "Fractional anisotropy": "FA",
        "Mean diffusivity": "MD",
        "Parallel diffusivity": "AD",
        "Perpendicular diffusivity": "RD",
        "Isotropic restricted water fraction": "IRF",
        "Free water fraction": "IFW"
    }

    # Read the VTK PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read VTK file or it's empty: {vtk_file_path}")
        return output_nifti_paths

    # Read the reference NIfTI image
    try:
        ref_image = nib.load(reference_nifti_path)
        ref_data = ref_image.get_fdata()
        ref_affine = ref_image.affine
        ref_shape = ref_data.shape[:3]  # Ensure we get only spatial dimensions
    except Exception as e:
        print(f"Error reading reference NIfTI file: {reference_nifti_path}")
        print(e)
        return output_nifti_paths

    # Read all arrays from the VTK file
    point_data = polydata.GetPointData()
    num_points = polydata.GetNumberOfPoints()
    num_arrays = point_data.GetNumberOfArrays()

    if num_arrays == 0:
        print(
            f"Error: No point data arrays found in VTK file: {vtk_file_path}")
        return output_nifti_paths

    # Extract VTK points and convert to physical coordinates
    vtk_points = np.array([polydata.GetPoint(i) for i in range(num_points)])

    # Create grid points in voxel space
    x, y, z = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing='ij'
    )

    grid_coords = np.vstack(
        [x.ravel(), y.ravel(), z.ravel(), np.ones_like(x.ravel())])
    world_coords = np.dot(ref_affine, grid_coords)[
        :3].T  # Transform and get (N, 3) shape

    for idx_array in range(num_arrays):
        array = point_data.GetArray(idx_array)
        array_name = array.GetName()
        if not array_name:
            array_name = f"UnnamedArray_{idx_array}"

        num_components = array.GetNumberOfComponents()

        # Skip arrays with multiple components for simplicity
        if num_components > 1:
            print(
                f"Skipping array '{array_name}' with {num_components} components")
            continue

        # Get scalar values from the VTK array
        vtk_values = vtk_to_numpy(array)
        #

        inv_ref_affine = np.linalg.inv(ref_affine)

        sum_fa_map = np.zeros(ref_shape, dtype=float)
        count_map = np.zeros(ref_shape, dtype=int)

        # Transformation de tous les points VTK en coordonnées voxel NIfTI
        # Ajout de la coordonnée homogène aux points VTK
        vtk_points_homogeneous = np.hstack(
            (vtk_points, np.ones((vtk_points.shape[0], 1))))
        # Transformation en coordonnées voxel
        vtk_points_voxel_coords_float = np.dot(
            inv_ref_affine, vtk_points_homogeneous.T)[:3, :].T

        # Itérer sur chaque point VTK et sa valeur de FA
        for i in range(vtk_points.shape[0]):
            point_coord_voxel_float = vtk_points_voxel_coords_float[i]
            fa_value = vtk_values[i]

            # Convertir les coordonnées voxel flottantes en indices entiers
            # et s'assurer qu'ils sont dans les limites de l'image
            vx = int(round(point_coord_voxel_float[0]))
            vy = int(round(point_coord_voxel_float[1]))
            vz = int(round(point_coord_voxel_float[2]))

            if (0 <= vx < ref_shape[0] and
                0 <= vy < ref_shape[1] and
                    0 <= vz < ref_shape[2]):

                sum_fa_map[vx, vy, vz] += fa_value
                count_map[vx, vy, vz] += 1

        # Calculer la carte de FA moyenne
        # Éviter la division par zéro pour les voxels sans points
        mean_fa_map = np.zeros(ref_shape, dtype=float)
        # Utiliser np.true_divide pour gérer la division par zéro et obtenir des NaN si souhaité,
        # ou vérifier count_map > 0 pour affecter 0 aux voxels vides.
        non_zero_counts = count_map > 0
        mean_fa_map[non_zero_counts] = sum_fa_map[non_zero_counts] / \
            count_map[non_zero_counts]

        # Sauvegarder la carte de FA moyenne
        output_mean_fa_path = os.path.join(
            output_dir, f"{output_file_prefix}_{name_mapping[array_name]}.nii.gz")
        mean_fa_img = nib.Nifti1Image(mean_fa_map, ref_affine)
        nib.save(mean_fa_img, output_mean_fa_path)
        print(f"Generated Mean FA in Voxel NIfTI file: {output_mean_fa_path}")
        output_nifti_paths.append(output_mean_fa_path)

    return output_nifti_paths


def add_mcm_to_tracts(tracts, mcm_file, reference=None, **kwargs):
    """
    Add MCM information to tracts.

    Parameters
    ----------
    tracts : ActiDepFile
        The tracts to which MCM information will be added
    mcm_file : ActiDepFile
        The MCM file containing the compartment information (.mcmx)
    reference : ActiDepFile, optional
        Reference image for tractogram conversion if needed.
    kwargs : dict
        Additional arguments for run_cli_command.
    """
    print("Adding MCM to tracts...")
    inputs = {"tracts": tracts, "mcm_file": mcm_file}
    if reference:
        # Ensure reference is available if needed by commands run by run_cli_command
        inputs["reference"] = reference

    tracts_extension = os.path.basename(tracts.path).split('.')[-1].lower()

    temp_dir_for_conversion = None  # To store path to temp directory if created
    converted_input_vtk_path = tracts.path

    mcm_obj = MCMFile(mcm_file.path)


    if tracts_extension != 'vtk':
        if reference is None:
            raise ValueError(
                "Reference image is required for tractogram conversion.")

        temp_dir_for_conversion = tempfile.mkdtemp(suffix='tracts_conversion')

        print("Temporary directory for conversion:", temp_dir_for_conversion, os.path.exists(temp_dir_for_conversion))
        os.chdir(temp_dir_for_conversion)
        
        base_name = os.path.basename(tracts.path)
        vtk_name = os.path.splitext(base_name)[0] + '.vtk'
        converted_input_vtk_path = opj(temp_dir_for_conversion, vtk_name)

        mcm_weights = mcm_obj.get_weights()
        print("MCM Weights file:", mcm_weights)

        cmd = [
            "flip_tractogram", tracts.path, converted_input_vtk_path, "--reference",
            reference.path
        ]
        print("Converting tracts to VTK format...")
        print("Command:", ' '.join(cmd))
        call(cmd)

        #Now call flip_vtk
        cmd = [
            "flip_vtk", converted_input_vtk_path, converted_input_vtk_path,'--compartment_map_file',mcm_weights
        ]
        print("Flipping VTK tracts...")
        print("Command:", ' '.join(cmd))
        call(cmd)


    # Output filename for animaTracksMCMPropertiesExtraction, relative to its execution directory
    output_vtk_filename_in_tmp = "tracts_mcm_out.vtk"

    command_args = [
        # Path to VTK (original or temp converted)
        "-i", converted_input_vtk_path,
        "-m", mcm_obj.mcmfile,          # Path to .mcm file
        "-o", output_vtk_filename_in_tmp
    ]

    base_entities = tracts.get_full_entities()

    # Define the expected VTK output pattern for run_cli_command
    output_pattern = {output_vtk_filename_in_tmp:
                      upt_dict(base_entities, {
                          "model": "MCM", 'extension': "vtk"})}

    # Run the anima command
    cli_results = run_cli_command(
        "animaTracksMCMPropertiesExtraction",
        inputs,
        output_pattern,
        tracts,  # Base ActiDepFile for deriving output metadata
        command_args=command_args,
        **kwargs
    )

    # # Cleanup the temporary directory used for VTK conversion
    # if temp_dir_for_conversion:
    #     shutil.rmtree(temp_dir_for_conversion)

    if not cli_results:
        print("Error: animaTracksMCMPropertiesExtraction failed or produced no output.")
        return {}

    # Get the absolute path of the generated VTK file from cli_results
    # Assuming cli_results has one entry for the VTK file
    generated_vtk_path = next(iter(cli_results.keys()))
    # # Prepare the final result dictionary, starting with the VTK file result
    final_results = cli_results.copy()

    # # Convert the output VTK to CSVs
    # # Base path for CSV names is the VTK path without extension
    # csv_output_base_path = os.path.splitext(generated_vtk_path)[0]

    # csv_files_dict = convert_vtk_to_csv(
    #     generated_vtk_path, csv_output_base_path)

    # # Add CSV files to the result dictionary
    # # Use original tracts for base entities
    # base_entities_for_csv = tracts.get_entities()

    # for content_key, csv_path_val in csv_files_dict.items():
    #     final_results[csv_path_val] = upt_dict(
    #         base_entities_for_csv.copy(),
    #         model="MCM",
    #         extension="csv",
    #         desc=content_key  # Store array name or 'fibers' in 'desc' field
    #     )

    # Generate scalar maps
    if reference:
        output_directory_for_nifti = os.path.dirname(generated_vtk_path)
        # Prefix for NIfTI files, e.g., "tracts_mcm_out" from "tracts_mcm_out.vtk"
        nifti_file_prefix = os.path.splitext(
            os.path.basename(generated_vtk_path))[0]

        generated_nifti_paths = generate_scalar_map(  # Calling the new multi-output function
            vtk_file_path=generated_vtk_path,
            reference_nifti_path=reference.path,
            output_dir=output_directory_for_nifti,
            output_file_prefix=nifti_file_prefix
        )

        # Remove the pipeline
        print("Base entities for NIfTI:", base_entities)
        for nifti_path in generated_nifti_paths:
            # e.g., "tracts_mcm_out_FA.nii.gz"
            filename_base = os.path.basename(nifti_path)

            # Extract array name from filename for the 'desc' field
            temp_name = filename_base
            if temp_name.endswith(".nii.gz"):
                temp_name = temp_name[:-len(".nii.gz")]
            # temp_name is now, e.g., "tracts_mcm_out_FA"

            array_desc_name = "unknown_array"
            if temp_name.startswith(nifti_file_prefix + "_"):
                array_desc_name = temp_name[len(
                    nifti_file_prefix) + 1:]

            final_results[nifti_path] = upt_dict(
                base_entities.copy(),
                model="MCM",
                extension="nii.gz",
                metric=f"{array_desc_name}"
            )

    pprint(final_results)
    return final_results


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

    # mcmfile = MCMFile('/tmp/tmpq4y0e7ei/test/mcm.mcm')
    # mcmfile.write('/tmp/tmpq4y0e7ei/test.mcmx')
    # mcmfile = MCMFile('/tmp/tmpq4y0e7ei/test.mcmx')
    # print(mcmfile.compartments)

    mcm_vtk = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors_staniz/sub-01002/tracto/sub-01002_desc-cleaned_model-MCM_tracto.vtk"

    fa_nii = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/anima_preproc/sub-01002/dwi/sub-01002_metric-FA_model-DTI_dwi.nii.gz"

    generate_scalar_map(
        vtk_file_path=mcm_vtk,
        reference_nifti_path=fa_nii,
        output_dir='test_scalar_map',
        output_file_prefix="tracts_mcm_out"
    )
