import os
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, Actidep
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, process_bundleseg, prepare_atlas_for_recobundle
from actiDep.utils.tractography import get_tractogram_endings
from actiDep.set_config import get_HCP_bundle_names
from actiDep.analysis.tractometry import process_projection
import multiprocessing
import tempfile
from glob import glob
import traceback
from subprocess import call
import json

HCP_CENTROIDS_DIR = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/flipped/centroids_frechet"
HCP_REFERENCE = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz"
HCP_BUNDLES = "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/"

def init_pipeline(subject, pipeline, **kwargs):
    """Initialize the MCM pipeline"""
    create_pipeline_description(
        pipeline,
        layout=subject.layout,
        **kwargs
    )
    return True

def segment_bundle_in_atlas_space(subject, pipeline, bundle,atlas_name='HCP105Group1Clustered', **kwargs):
    """
    Segment a bundle from a whole brain tractography

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle : str
        Name of the bundle to segment
    """
    # Load the whole brain tractography
    whole_brain_tract = subject.get_unique(suffix='tracto', pipeline=pipeline, label='brain',extension='trk', space=atlas_name)

    # Load the bundle template
    # template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/695768/tracts/{bundle}.trk"
    template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_{bundle}.tck"

    # Register the bundle template to the subject
    bundle_tract = call_recobundle(whole_brain_tract, template_path,atlas_name=atlas_name,bundle_name=bundle,**kwargs)

    # Save the segmented bundle
    copy_from_dict(subject, bundle_tract, pipeline=pipeline, bundle=bundle, desc='noslr')
    return True

def run_bundleseg(subject, pipeline, atlas_dir='/home/ndecaux/Data/Atlas',config='config.json', **kwargs):
    """
    Run the bundlesegmentation pipeline on the given subject.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    pipeline : str
        Pipeline name
    bundle_list : list of str, optional
        List of bundles to segment. If None, all bundles will be segmented.
    atlas_name : str, optional
        Name of the atlas to use for segmentation. Default is 'SCIL'.
    """

    # Load the whole brain tractography
    anat = subject.get(metric='FA', pipeline='anima_preproc', extension='nii.gz')[0]
    tracto = subject.get(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')[0]
    
    # Register the anatomical image to the template space
    res_dict=process_bundleseg(tracto, anat, atlas_dir=atlas_dir,config=config)
    copy_from_dict(subject,res_dict,pipeline=pipeline)

    

def run_bundleseg_autocalibration(subject,pipeline='bundle_seg',**kwargs):
    """
    Run the bundlesegmentation pipeline on the given subject with autocalibration.
    
    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    pipeline : str
        Pipeline name
    bundle_list : list of str, optional
        List of bundles to segment. If None, all bundles will be segmented.
    atlas_name : str, optional
        Name of the atlas to use for segmentation. Default is 'SCIL'.
    """

    # Load the whole brain tractography
    anat = subject.get_unique(metric='FA', pipeline='anima_preproc', extension='nii.gz')
    pseudo_models = subject.get(suffix='tracto', pipeline=pipeline, extension='trk')
    atlas_dir=prepare_atlas_for_recobundle([t.path for t in pseudo_models], model_config = 8, model_T1=anat.path)

    print('Atlas directory prepared at:', atlas_dir)
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')

    res_dict=process_bundleseg(tracto, anat, atlas_dir=atlas_dir)
    copy_from_dict(subject,res_dict,pipeline=pipeline+'_autocalib')


def run_bundle_seg_selected_bundles(subject, pipeline, bundle_list, **kwargs):
    """
    Run the bundlesegmentation pipeline on the given subject for selected bundles.

    Parameters
    ----------
    subject : str or Subject
        Subject ID or Subject object to process
    pipeline : str
        Pipeline name
    bundle_list : list of str
        List of bundles to segment.
    """

    if isinstance(subject, str):
        subject = Subject(subject)
    model_files = glob(f'{HCP_BUNDLES}/summed_*.trk')
    model_files = [f for f in model_files if any([get_HCP_bundle_names(b) in f for b in bundle_list])]

    atlas_dir=prepare_atlas_for_recobundle(model_files, model_config = 12, model_T1=HCP_REFERENCE)

    print('Atlas directory prepared at:', atlas_dir)
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')
    anat_subject=subject.get_unique(metric='FA', pipeline='anima_preproc', extension='nii.gz')
    res_dict=process_bundleseg(tracto, anat_subject, atlas_dir=atlas_dir)
    copy_from_dict(subject,res_dict,pipeline=pipeline)


def run_bundleseg_on_centroids(subject, pipeline, **kwargs):
    anat = HCP_REFERENCE
    # models_vtk = glob(f'{HCP_CENTROIDS_DIR}/*_centroids.vtk')
    models = glob(f'{HCP_CENTROIDS_DIR}/*_centroids.trk')
    print(f'Found {len(models)} centroid models')
    # if len(models) == 0 or len(models_vtk) != 0 :
    #     #Convert all vtk to trk
    #     for vtk in models_vtk:
    #         trk = vtk.replace('.vtk', '.trk')
    #         cmd=f'flip_tractogram --reference {anat} {vtk} {trk}'
    #         call(cmd, shell=True)
    #         models.append(trk)
            

    atlas_dir = prepare_atlas_for_recobundle(models, model_config = 60, model_T1=anat)
    print('Atlas directory prepared at:', atlas_dir)
    
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck')
    anat_subject=subject.get_unique(metric='FA', pipeline='anima_preproc', extension='nii.gz')
    res_dict=process_bundleseg(tracto, anat_subject, atlas_dir=atlas_dir)
    copy_from_dict(subject,res_dict,pipeline=pipeline+'_centroids')

def copy_bundleseg_result(subject, result_folder, pipeline, **kwargs):
    """
    Copy the bundlesegmentation result to the subject's directory.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_name : str
        Name of the bundle to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    sub_id = subject.sub_id
    # get the folder in result_folder that contains <sub_id>_<SOMETHING>
    sub_folder = [x for x in os.listdir(result_folder) if x.startswith(sub_id+'_')][0]
    #Get the subsubfolder in sub_folder that contains files with _cleaned.trk
    trk_paths = glob(f'{os.path.join(result_folder, sub_folder)}/*/*/*_cleaned.trk')
    bundle_name_mapping = {v:k for k, v in get_HCP_bundle_names().items()}
    
    entities= subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',algo='ifod2',extension='tck').get_full_entities()
    entities['extension'] = 'trk'


    res_dict = {f:upt_dict(entities,{'bundle': bundle_name_mapping[f.split('summed_')[-1].split('_cleaned')[0]]}) for f in trk_paths}

    copy_from_dict(subject,res_dict,pipeline=pipeline)
    
def get_single_bundleseg_endings(subject, pipeline, bundle_name, **kwargs):
    """
    Get the endings segmentation of the streamlines in the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    bundle_name : str
        Name of the bundle to segment
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    tracto = subject.get_unique(suffix='tracto', pipeline=pipeline, bundle=bundle_name,extension='trk')

    # Get the endings segmentation
    endings = get_tractogram_endings(tracto, reference=subject.get_unique(metric='FA', pipeline='anima_preproc', extension='nii.gz'))

    # Save the endings segmentation
    copy_from_dict(subject, endings, pipeline=pipeline, bundle=bundle_name, suffix='mask', datatype='endings')


def get_bundleseg_endings(subject, pipeline):
    """
    call get_single_bundleseg_endings for all bundles in the bundlesegmentation tractography
    """

    bundle_list = subject.get(suffix='tracto',
                              pipeline=pipeline,
                              extension='trk')
    print(bundle_list)
    already_done = subject.get(suffix='mask',
                               pipeline=pipeline,
                               desc='endings')
    already_done = [x.get_full_entities()['bundle'] for x in already_done]
    print(f"Already done: {already_done}")

    # print(f"Bundles to process: {bundle_list}")
    for bundle in bundle_list:
        try :
            bundle_name = bundle.get_full_entities()['bundle']
            if False:# bundle_name in already_done:
                print(f"Bundle {bundle_name} already processed, skipping")
                continue
            print(f"Processing bundle {bundle_name}")
            get_single_bundleseg_endings(subject, pipeline, bundle_name)
        except Exception as e:
            print(f"Error processing bundle {bundle}: {e}")
            continue
    return True


def project_metric_onto_bundleseg(subject, pipeline, metric_name, **kwargs):
    """
    Project a metric onto the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Subject
        Subject object to process
    pipeline : str
        Pipeline name
    metric_name : str
        Name of the metric to project
    kwargs : dict
        Additional keyword arguments for processing
    """

    # Load the bundlesegmentation tractography
    bundle_list = subject.get(suffix='tracto',
                              pipeline=pipeline,
                              extension='trk')

    # Load the metric to project
    metric_list = subject.get(pipeline='mcm_tensors_staniz',
                              extension='nii.gz',
                              metric=metric_name)
    metric_dict = {
        x.get_full_entities()['bundle']:x for x in metric_list
    }

    beginnings_dict = subject.get(suffix='mask', datatype='endings',
                                  label='start')
    beginnings_dict = {
        x.get_full_entities()['bundle']:x for x in beginnings_dict if x.get_full_entities()['bundle'] in metric_dict.keys()
    }

    #Remove bundle from bundle_list if not in metric_list
    bundle_dict = {
        x.get_full_entities()['bundle']:x for x in bundle_list
        if x.get_full_entities()['bundle'] in metric_dict.keys()
    }

    print(f'Starting projection of {len(bundle_dict)} bundles with metric {metric_name}')

    # Project the metric onto the bundlesegmentation tractography
    res_dict = process_projection(bundle_dict, metric_dict, beginnings_dict, **kwargs)

    copy_from_dict(subject,
                   res_dict,
                   pipeline=pipeline,
                   metric=metric_name,
                   desc='projected',
                   datatype='metric')
    

def list_missing_bundleseg(dataset, pipeline):
    """
    List the bundles that are missing in the bundlesegmentation tractography.

    Parameters
    ----------
    subject : Actidep
        Actidep dataset object
    pipeline : str
        Pipeline name
    """

    all_bundles = set(get_HCP_bundle_names().keys())
    missing_bundles = {}
    ds_bundles = dataset.get_global(suffix='tracto', pipeline=pipeline, extension='trk')
    for sub in dataset.subject_ids:
        sub_bundles = set([x.get_full_entities()['bundle'] for x in ds_bundles if x.get_full_entities()['subject'] == sub])
        missing = all_bundles - sub_bundles
        if len(missing) > 0:
            missing_bundles[sub] = list(missing)
            print(f"Subject {sub} is missing bundles: {missing}")
    #Save as json file
    with open(f'missing_bundles_{pipeline}.json', 'w') as f:
        json.dump(missing_bundles, f, indent=4)

    return missing_bundles

def segment_subject_bundleseg(subject,pipeline='recobundle_segmentation', **kwargs):
    """
    Process the MSMT-CSD pipeline on the given subject.

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
         'run_bundleseg',
        #  'run_bundleseg_on_SLF',
        # 'run_bundleseg_on_centroids',

        #  'run_bundleseg_autocalibration',
        # 'get_bundleseg_endings',
        # "project_metric_onto_bundleseg"
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'run_bundleseg': lambda: run_bundleseg(subject, pipeline, **kwargs),
        'run_bundleseg_on_SLF': lambda: run_bundleseg(subject, pipeline, atlas_dir='/home/ndecaux/Data/Atlas',config='config_SLF.json', **kwargs),
        'run_bundleseg_autocalibration': lambda: run_bundleseg_autocalibration(subject, pipeline, **kwargs),
        'run_bundleseg_on_centroids': lambda: run_bundleseg_on_centroids(subject, pipeline, **kwargs),
        'get_bundleseg_endings': lambda: get_bundleseg_endings(subject, pipeline, **kwargs),
        'project_metric_onto_bundleseg': lambda: project_metric_onto_bundleseg(subject, pipeline, metric_name='IFW', **kwargs),
    }

    for step in pipeline_list:
        if step in step_mapping:
            print(f"Running step: {step}")
            step_mapping[step]()
    
    return True


def process_single_subject(arg):
    """Process a single subject with the given arguments"""

    try :
        sub, dataset_path, pipeline = arg
        print(f"Processing subject {sub} with pipeline {pipeline}")
        subject = Subject(sub, db_root=dataset_path)
        return segment_subject_bundleseg(subject, pipeline=pipeline)
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

from pprint import pprint

if __name__ == "__main__":
    pipeline = 'bundle_seg'
    num_processes = 1

    if os.uname()[1] == 'calcarine':
        num_processes = 32
        print("calcarine")
        # tempfile.tempdir = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bundle_seg'
        tempfile.tempdir = '/local/ndecaux/bundle_seg'
        #also set the TMPDIR env variable
        os.environ['TMPDIR'] = tempfile.tempdir
    else:
        num_processes = 1
        print(f"Not calcarine, using {num_processes} processes")
        tempfile.tempdir = '/home/ndecaux/bundle_seg'
        os.environ['TMPDIR'] = tempfile.tempdir
    # else:
    #     #Tempdir on home
    #     tempfile.tempdir = os.path.join(os.path.expanduser('~'), 'bundle_seg')
    #     os.environ['TMPDIR'] = tempfile.tempdir


    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')

    # dataset_path = '/home/ndecaux/Code/Data/comascore'
    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    # dataset_path = '/home/ndecaux/Code/Data/dysdiago'
    # dataset_path='/home/ndecaux/NAS_EMPENN/share/projects/actidep/IRM_Cerveau_MOI/bids'
    ds = Actidep(dataset_path)
    # pprint(list_missing_bundleseg(ds, pipeline='bundle_seg_old'))

    pipeline='bundle_seg_old'
    missing_bundles = list_missing_bundleseg(ds, pipeline=pipeline)
    pprint(missing_bundles)

    for sub, bundles in missing_bundles.items():
        print(f"Processing subject {sub} with missing bundles: {bundles}")
        run_bundle_seg_selected_bundles(sub, pipeline=pipeline, bundle_list=bundles)

    # sub= Subject('00001',db_root=dataset_path)

    # run_bundleseg(sub, pipeline=pipeline, atlas_name='SCIL')


    # # sub = Subject('01002', db_root=dataset_path)

    # # project_metric_onto_bundleseg(sub, pipeline=pipeline, metric_name='FA')

    # subject_ids = ds.subject_ids[6:]
    # # subject_ids = ['03008']
    # args = [(sub, pipeline) for sub in subject_ids]
    # args_filtered = []
    # flag=False
    # for arg in args:
    #     sub, pipeline = arg
    #     if flag == False and sub == '03026':
    #         continue
    #     else:
    #         flag=True
        
    #     # sub = Subject(sub, db_root=dataset_path)
    #     args_filtered.append((sub, dataset_path, pipeline))

    # args = args_filtered
    # print(f"Found {len(args)} subjects to process")


    # # Définir le nombre de processus (ajustez selon les ressources disponibles)
    #multiprocessing.cpu_count() - 1  # Laisse un CPU libre

    # Pour exécuter en séquentiel (commentez les lignes multiprocessing ci-dessous)
    # for arg in args:
    #     process_single_subject(arg)

    # Exécution parallèle avec multiprocessing
    # print(f"Démarrage du traitement parallèle avec {num_processes} processus")
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(process_single_subject, args)

    # print("Traitement terminé pour tous les sujets")


    ## Copy the bundlesegmentation result to the subject's directory
    # for sub in ds.subject_ids:
    #     subject = Subject(sub, db_root=dataset_path)
    #     result_folder = "/local/ndecaux/BundleSegResults"
    #     copy_bundleseg_result(subject, result_folder, pipeline=pipeline)