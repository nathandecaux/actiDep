import os
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, Actidep
from actiDep.data.io import copy2nii, move2nii, copy_list, copy_from_dict
from actiDep.utils.tools import del_key, upt_dict, create_pipeline_description, CLIArg
from actiDep.utils.recobundle import register_template_to_subject, call_recobundle,register_anat_subject_to_template, process_bundleseg
from actiDep.utils.tractography import get_tractogram_endings
from actiDep.analysis.tractometry import process_projection
import multiprocessing
import dipy 

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import length
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa


def cluster_streamlines(tractogram, threshold=100.):
    tracto = load_tractogram(tractogram, "same", bbox_valid_check=False)

    #Resample to 100 points
    from dipy.tracking.streamline import set_number_of_points
    streamlines = set_number_of_points(tracto.streamlines, 100)
    
    #Save
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=100., metric=metric)
    clusters = qb.cluster(streamlines)
    print('N clusters',len(clusters))

    centroids = Streamlines(clusters.centroids)
    

if __name__ == "__main__":
    tracto = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/bundle_seg/sub-01002/tracto/sub-01002_bundle-CSTleft_desc-cleaned_tracto.trk"

    cluster_streamlines(tracto, threshold=100.)