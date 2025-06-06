import os
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, Actidep
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, process_bundleseg
from actiDep.utils.tractography import get_tractogram_endings
from actiDep.analysis.tractometry import process_projection
import multiprocessing


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
    template_path = f"/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_{bundle}.trk"

    # Register the bundle template to the subject
    bundle_tract = call_recobundle(whole_brain_tract, template_path,atlas_name=atlas_name,bundle_name=bundle,**kwargs)

    # Save the segmented bundle
    copy_from_dict(subject, bundle_tract, pipeline=pipeline, bundle=bundle, desc='noslr')
    return True

def run_bundleseg(subject, pipeline, atlas_name='SCIL', **kwargs):
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
    anat = subject.get_unique(metric='FA', pipeline='anima_preproc', extension='nii.gz')
    tracto = subject.get_unique(suffix='tracto', pipeline='msmt_csd', label='brain',desc='normalized',algo='ifod2',extension='tck')
    print(tracto)
    # Register the anatomical image to the template space
    process_bundleseg(tracto, anat, atlas_dir='/home/ndecaux/Data/Atlas')

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
                              desc='cleaned',
                              extension='trk')
    already_done = subject.get(suffix='mask',
                               pipeline=pipeline,
                               desc='endings')
    already_done = [x.get_full_entities()['bundle'] for x in already_done]
    print(f"Already done: {already_done}")

    # print(f"Bundles to process: {bundle_list}")
    for bundle in bundle_list:
        try :
            bundle_name = bundle.get_full_entities()['bundle']
            if bundle_name in already_done:
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
                              desc='cleaned',
                              extension='trk')

    # Load the metric to project
    metric_list = subject.get(pipeline='mcm_tensors_staniz',
                              extension='nii.gz',
                              metric=metric_name)
    metric_dict = {
        x.get_full_entities()['bundle']:x for x in metric_list
    }

    #Remove bundle from bundle_list if not in metric_list
    bundle_dict = {
        x.get_full_entities()['bundle']:x for x in bundle_list
        if x.get_full_entities()['bundle'] in metric_dict.keys()
    }

    print(f'Starting projection of {len(bundle_dict)} bundles with metric {metric_name}')

    # Project the metric onto the bundlesegmentation tractography
    res_dict = process_projection(bundle_dict, metric_dict, **kwargs)

    copy_from_dict(subject,
                   res_dict,
                   pipeline=pipeline,
                   metric=metric_name,
                   desc='projected',
                   datatype='metric')

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
        # 'run_bundleseg',
        # 'get_bundleseg_endings',
        "project_metric_onto_bundleseg"
    ]

    # Process each requested pipeline step
    step_mapping = {
        'init': lambda: init_pipeline(subject, pipeline),
        'run_bundleseg': lambda: run_bundleseg(subject, pipeline, atlas_name='SCIL', **kwargs),
        'get_bundleseg_endings': lambda: get_bundleseg_endings(subject, pipeline, bundle_name='IFOD2', **kwargs),
        'project_metric_onto_bundleseg': lambda: project_metric_onto_bundleseg(subject, pipeline, metric_name='AD', **kwargs),
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
        return False


if __name__ == "__main__":
    pipeline = 'bundle_seg'
    # if os.uname()[1] == 'calcarine':
    #     print("calcarine")
    #     # tempfile.tempdir = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bundle_seg'
    #     tempfile.tempdir = '/local/ndecaux/bundle_seg'
    #     #also set the TMPDIR env variable
    #     os.environ['TMPDIR'] = tempfile.tempdir
    # else:
    #     #Tempdir on home
    #     tempfile.tempdir = os.path.join(os.path.expanduser('~'), 'bundle_seg')
    #     os.environ['TMPDIR'] = tempfile.tempdir


    config, tools = set_config()
    # subject = Subject('100206',db_root='/home/ndecaux/Data/HCP/')

    dataset_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'

    ds = Actidep(dataset_path)

    # sub = Subject('01002', db_root=dataset_path)

    # project_metric_onto_bundleseg(sub, pipeline=pipeline, metric_name='FA')

    args = [(sub, pipeline) for sub in ds.subject_ids]
    args_filtered = []
    flag=False
    for arg in args:
        sub, pipeline = arg
        if flag == False and sub == '03026':
            continue
        else:
            flag=True
        
        # sub = Subject(sub, db_root=dataset_path)
        args_filtered.append((sub, dataset_path, pipeline))

    args = args_filtered
    print(f"Found {len(args)} subjects to process")


    # Définir le nombre de processus (ajustez selon les ressources disponibles)
    num_processes = 40#multiprocessing.cpu_count() - 1  # Laisse un CPU libre

    # Pour exécuter en séquentiel (commentez les lignes multiprocessing ci-dessous)
    # for arg in args:
    #     process_single_subject(arg)

    # Exécution parallèle avec multiprocessing
    print(f"Démarrage du traitement parallèle avec {num_processes} processus")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_subject, args)

    print("Traitement terminé pour tous les sujets")
