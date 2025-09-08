import os 
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, Actidep
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.clustering import associate_subject_to_centroids
from actiDep.set_config import get_HCP_bundle_names
from dipy.tracking.streamline import set_number_of_points
import tempfile
import glob
import shutil
from time import sleep
import time
import multiprocessing

# Répertoire contenant les centroids HCP
HCP_CENTROIDS_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/long_central_line"
HCP_FULL_BUNDLE_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/vtk/"
HCP_REFERENCE = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz"

pipeline='hcp_association_24pts'
# def get_bundle_mapping():
#     """
#     Créer un mapping entre les noms de bundles et les fichiers centroids correspondants.
    
#     Returns
#     -------
#     bundle_mapping : dict
#         Dictionnaire associant les noms de bundles aux chemins des centroids
#     """
#     bundle_mapping = {}
    
#     # Scanner les fichiers centroids disponibles
#     centroid_files = glob.glob(os.path.join(HCP_CENTROIDS_DIR, "*_centroids.vtk"))
    
#     for centroid_file in centroid_files:
#         filename = os.path.basename(centroid_file)
#         # Extraire le nom du bundle (ex: summed_CST_left_centroids.vtk -> CST_left)
#         bundle_name = filename.replace('summed_', '').replace('_centroids.vtk', '')
        
#         actidep_name = [k for k,v in get_HCP_bundle_names().items() if v==bundle_name]
        
#         bundle_mapping[actidep_name] = centroid_file
    
#     print(f"Bundle mapping créé : {bundle_mapping}")
#     return bundle_mapping

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the HCP association pipeline"""
    if not isinstance(subject, Subject):
        subject = Subject(subject)
    print(subject)
    create_pipeline_description(
        pipeline, 
        layout=subject.layout,  # Utiliser subject.layout comme prévu
        registrationMethod='SLR', 
        registrationParams='streamline_linear_registration', 
        exactParams={'model': 'HCP_centroids'},
        **kwargs
    )
    return True

def process_bundle_associations(subject, pipeline, **kwargs):
    """
    Traiter toutes les associations de bundles pour un sujet.
    
    Parameters
    ----------
    subject : Subject
        Objet Subject à traiter
    pipeline : str
        Nom du pipeline
    """
    # Obtenir le mapping des bundles
    bundle_mapping = get_HCP_bundle_names()
    
    # Obtenir tous les fichiers VTK de la pipeline mcm_to_hcp_space
    vtk_files = subject.get(
        pipeline='mcm_to_hcp_space',
        space='HCP', 
        extension='vtk',
        datatype='tracto'
    )
    #Sort les fichiers par nom de bundle
    vtk_files.sort(key=lambda x: x.get_entities().get('bundle', ''))
    


    print(f"Trouvé {len(vtk_files)} fichiers VTK pour le sujet {subject.sub_id}")
    # vtk_files = [vtk_file for vtk_file in vtk_files if vtk_file.get_entities().get('bundle', '') == 'CC1']
    for vtk_file in vtk_files:
        # Extraire le nom du bundle depuis les entités
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        
        if bundle_name in bundle_mapping:
            hcp_bundle_name = bundle_mapping[bundle_name]
            centroid_path = os.path.join(HCP_CENTROIDS_DIR, f'summed_{hcp_bundle_name}_centroids.vtk')
            
            model_bundle_path = os.path.join(HCP_FULL_BUNDLE_DIR, f'summed_{hcp_bundle_name}.vtk')
            print(f"Traitement du bundle {bundle_name} avec centroids {centroid_path}")
            
            try:
                # Appliquer l'association
                start_time = time.time()
                res_dict = associate_subject_to_centroids(
                    subject_bundle=vtk_file,
                    model_centroids_path=centroid_path,
                    model_full_bundle_path=model_bundle_path,
                    reference_nifti=HCP_REFERENCE,
                    **kwargs
                )
                elapsed_time = time.time() - start_time
                print(f"associate_subject_to_centroids executed in {elapsed_time:.2f} seconds")
                
                # Copier les résultats dans le layout BIDS
                copy_from_dict(subject, res_dict, pipeline=pipeline)
                
                print(f"Association terminée pour {bundle_name}")
                
            except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
                import traceback
                print(f"Erreur lors du traitement de {bundle_name}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
        else:
            print(f"Pas de centroids trouvés pour le bundle {bundle_name}")

def combine_csv(subject, pipeline):
    """
    Combine les fichiers CSV de métriques pour un sujet donné dans le format utilisé par Tractseg : un csv par métrique, contenant les moyennes pour chaque bundle (une colonne par bundle).
    """
    if not isinstance(subject, Subject):
        subject = Subject(subject)
    # Obtenir tous les fichiers CSV de métriques
    csv_files = subject.get(
        pipeline=pipeline,
        extension='csv',
        suffix='mean',
        datatype='metric'
    )
    if not csv_files:
        print(f"Aucun fichier CSV trouvé pour le sujet {subject.sub_id} dans le pipeline {pipeline}")
        return
    print(f"Trouvé {len(csv_files)} fichiers CSV pour le sujet {subject.sub_id} dans le pipeline {pipeline}")

    temp_dir = tempfile.mkdtemp()

    csv_dfs = [(csv_file.get_entities().get('bundle', ''), pd.read_csv(csv_file.path, delimiter=',')) for csv_file in csv_files]
    # Créer un dictionnaire pour stocker les DataFrames
    df_dict = {}
    entities= csv_files[0].get_entities()

    for metric in ['AD','RD','MD','FA','IFW','IRF']:
        df = pd.DataFrame()
        for bundle_name, bundle_df in csv_dfs:
            df[bundle_name] = bundle_df[metric]
        
        #Save the DataFrame to a CSV file in a temporary directory
        output_path = os.path.join(temp_dir, f"{metric}_metrics.csv")
        #set nan to 0.0
        df.fillna(0.0, inplace=True)
        df.to_csv(output_path, index=False,sep=';')

        entities_metric = entities.copy()

        entities_metric['metric'] = metric
        del entities_metric['bundle']
        df_dict[output_path] = entities_metric
        print(f"Fichiers CSV combinés enregistrés dans {output_path}")

    # Copier les fichiers CSV combinés dans le layout BIDS
    copy_from_dict(
        subject=subject,
        file_dict=df_dict,
        pipeline='hcp_association_tractseg',
        datatype='metric'
    )

def get_central_line_displacement(subject, pipeline=pipeline):
    """
    Compute the MDF between the subject's bundles after HCP association and the HCP centroids.
    Parameters
    ----------
    subject : Subject
        Subject object to process
    
    pipeline : str
        Pipeline name
    Returns
    -------
    CSV file path
        Path to the CSV file containing the MDF values for each bundle
    """

    if not isinstance(subject, Subject):
        subject = Subject(subject)
    
    # Obtenir le mapping des bundles
    bundle_mapping = get_HCP_bundle_names()
    
    # Obtenir tous les fichiers VTK de la pipeline hcp_association
    vtk_files = subject.get(
        pipeline=pipeline,
        desc='centroids',
        extension='vtk',
        datatype='tracto'
    )
    vtk_files.sort(key=lambda x: x.get_entities().get('bundle', ''))
    
    mdf_values = {}
    
    for vtk_file in vtk_files:
        entities = vtk_file.get_entities()
        bundle_name = entities.get('bundle', '')
        
        if bundle_name in bundle_mapping:
            hcp_bundle_name = bundle_mapping[bundle_name]
            centroid_path = os.path.join(HCP_CENTROIDS_DIR, f'summed_{hcp_bundle_name}_centroids.vtk')
            
            try:
                from dipy.tracking.distances import bundles_distances_mdf
                from dipy.io.streamline import load_vtk_streamlines
                
                # Charger les streamlines
                subject_streamlines = load_vtk_streamlines(vtk_file.path)
                centroid_streamlines = load_vtk_streamlines(centroid_path)
                
                subject_streamlines = set_number_of_points(subject_streamlines, 24)
                centroid_streamlines = set_number_of_points(centroid_streamlines, 24)
                if len(subject_streamlines) == 0 or len(centroid_streamlines) == 0:
                    print(f"Streamlines vides pour le bundle {bundle_name}, skipping MDF calculation.")
                    continue
                
                # Calculer le MDF
                mdf = bundles_distances_mdf(subject_streamlines, centroid_streamlines)
                mean_mdf = np.mean(mdf)
                
                mdf_values[bundle_name] = mean_mdf
                print(f"MDF for {bundle_name}: {mean_mdf}")
                
            except Exception as e:
                print(f"Erreur lors du calcul du MDF pour {bundle_name}: {e}")
                continue
        else:
            print(f"Pas de centroids trouvés pour le bundle {bundle_name}")
    
    # Sauvegarder les valeurs MDF dans un fichier CSV
    if mdf_values:
        entities = {'pipeline': pipeline, 'desc': 'hcpdisp','datatype':'stats', 'extension': 'csv',"suffix":'mdf'}
        output_path = subject.build_path(**entities)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(list(mdf_values.items()), columns=['bundle', 'MDF'])
        df.to_csv(output_path, index=False, sep=';')
        print(f"MDF values saved to {output_path}")
        return output_path

    

def process_hcp_association(subject,pipeline=pipeline):
    """
    Process the HCP association pipeline on the given subject.
    
    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    """

    if isinstance(subject, str):
        subject = Subject(subject)

    # Define processing steps
    pipeline_list = [
        # 'init',
        # 'bundle_associations',
        'combine_csv',
        # 'central_line_displacement'
    ]
    
    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'bundle_associations': lambda: process_bundle_associations(subject, pipeline),
        'combine_csv': lambda: combine_csv(subject, pipeline),
        'central_line_displacement': lambda: get_central_line_displacement(subject, pipeline)
    }
    
    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
            # Refresh the subject object to ensure it has the latest data
            subject = Subject(subject.sub_id, db_root=subject.db_root)

if __name__ == "__main__":
    # If hostname is calcarine, set tempdir to /local/ndecaux/tmp
    if os.uname()[1] == 'calcarine':
        tempfile.tempdir = '/local/ndecaux/tmp'

    config, tools = set_config()
    print("HCP Association pipeline")
    print("=====================================")
    print('Reading dataset')
    # db_root = '/home/ndecaux/Code/Data/dysdiago'
    db_root = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    ds = Actidep(db_root)
    print(f"Found {len(ds.subject_ids)} subjects")
    print("=====================================")
    
    def process_one_subject(sub):
        try:
            subject = ds.get_subject(sub)
            # Vérifier si le sujet a des fichiers VTK dans mcm_to_hcp_space
            vtk_files = subject.get(
                pipeline='mcm_to_hcp_space',
                space='HCP', 
                extension='vtk',
                datatype='tracto'
            )
            # if len(vtk_files) == 0:
            #     print(f"Skipping subject {sub} - pas de fichiers VTK dans mcm_to_hcp_space")
            #     return
            # Vérifier si les associations existent déjà
            existing_associations = subject.get(
                pipeline=pipeline,
                desc='associations',
                extension='vtk'
            )
            # if len(existing_associations) > 70:
            #     print(f"Skipping subject {sub} - associations déjà existantes")
            #     return
            print(f"Processing subject: {sub}")
            process_hcp_association(subject)
        except Exception as e:
            print(f"Erreur lors du traitement du sujet {sub}: {e}")
            import traceback
            traceback.print_exc()
            return

    # Use multiprocessing to process subjects in parallel
    with multiprocessing.Pool(16) as pool:
        pool.map(process_one_subject, ds.subject_ids)
    print("HCP Association pipeline completed")

