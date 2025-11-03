from actiDep.data.loader import Actidep, Subject

sub=Subject('01001')
tracto_old = sub.get(pipeline='bundle_seg_old', extension='trk')
tracto_new = sub.get(pipeline='bundle_seg', extension='trk')
print("Old tractography:",len(tracto_old))
print("New tractography:",len(tracto_new))

from tractviewer import TractViewer

for b in tracto_new:
    vis = TractViewer(background="white", off_screen=False)
    bundle_name=b.get_entities()['bundle']
    print(bundle_name)
    vis.add_dataset(b.path, color='green', opacity=0.5)
    b_original = [bo for bo in tracto_old if bo.get_entities()['bundle'] == bundle_name]
    if len(b_original) == 1:
        vis.add_dataset(b_original[0].path, name=bundle_name + '_original', color='red', opacity=0.5)
    vis.record_rotation("bundle_seg_compare/rotation_{}.mp4".format(bundle_name), n_frames=240, step=1.5, quality=10)