import os
from actiDep.data.loader import Subject
from tractviewer import TractViewer
from actiDep.set_config import get_HCP_bundle_names
import multiprocessing

bundles=list(get_HCP_bundle_names().keys())
sub=Subject('01032')
ref_anat=sub.get_unique(pipeline='anima_preproc',metric='FA',extension='nii.gz')
tracto = sub.get(pipeline='filtered_tracto',extension='trk')

ref_anat=ref_anat.path

def process_bundle(b):
    if os.path.exists(f"out/{b}.gif"):
        print(f"Bundle {b} already processed, skipping.")
        return
    print(f"Displaying bundle {b}")
    b_tracto = [t for t in tracto if t.get_entities().get('bundle', None) == b]
    print([b.path for b in b_tracto])
    if len(b_tracto) != 2:
        print(f"No streamlines found for bundle {b}")
        return

    trekker = [b for b in b_tracto if b.get_entities().get('algo', None) == 'trekker'][0]
    ifod2 = [b for b in b_tracto if b.get_entities().get('algo', None) == 'ifod2'][0]

    vis = TractViewer(background="white", off_screen=True)

    vis.add_dataset(trekker.path, cmap='CET_L4', name='trekker', opacity=0.5)
    vis.add_dataset(ifod2.path, cmap='CET_L5', name='ifod2', opacity=0.5)
    hcp = sub.get_unique(pipeline='bundle_seg_nonrigid', bundle=b, atlas='HCP', extension='trk')
    vis.add_dataset(hcp.path, cmap='CET_L6', name='HCP', opacity=0.05)

    endings = sub.get(pipeline='bundle_seg_nonrigid', bundle=b, datatype='endings', extension='nii.gz')
    vis.add_dataset(endings[0].path, name='ending', opacity=0.05)
    vis.add_dataset(endings[1].path, name='start', opacity=0.05)

    vis.record_rotation(
        f"out/{b}.gif",
        n_frames=120,
        step=3,
        gif=True,
        fps=20
    )

if __name__ == "__main__":
    with multiprocessing.Pool(32) as pool:
        pool.map(process_bundle, bundles)
