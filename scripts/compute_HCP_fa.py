import os
from subprocess import call
atlas_dir="/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat"
#get home dir
DTI_folder = "/local/tdurante/atlas/DTI_Zenodo"

sub_list = [(d.split('.')[0].split('_')[-1],d) for d in os.listdir(DTI_folder) if d.endswith('.nii.gz')]

print(sub_list)

for sub,dti_file in sub_list:
    dti_path = os.path.join(DTI_folder,dti_file)
    out_fa = os.path.join(atlas_dir,sub,'dti',f"FA_{sub}.nii.gz")
    os.makedirs(os.path.dirname(out_fa),exist_ok=True)
    if os.path.exists(out_fa):
        print(f"FA already computed for subject {sub}, skipping")
        continue
    cmd = 'animaDTIScalarMaps -i {} -f {}'.format(dti_path, out_fa)
    print(cmd)
    call(cmd,shell=True)
