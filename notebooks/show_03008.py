from tractviewer import TractViewer
from actiDep.data.loader import Subject
sub = '03008'
sub=Subject(sub, db_root="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids")

tracto = sub.get(suffix='tracto', desc='associations',pipeline='hcp_association_24pts', extension='vtk')
print(len(tracto))

vis = TractViewer(background="white", off_screen=False)
color_dict = {}

for t in tracto:
    print(t.path)
    # Use a large palette and match left/right bundles to the same color
    import seaborn as sns
    # Use 'husl' palette for up to 100 distinct colors
    palette = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 16)
    bundle_name = t.get_entities().get('bundle')
    # Remove 'left'/'right' to match pairs
    base_name = bundle_name.replace('left', '').replace('right', '')
    if base_name not in color_dict:
        color_dict[base_name] = palette[len(color_dict) % len(palette)]
    color = color_dict[base_name]
    vis.add_dataset(
        t.path,
        {
            "color": color,
            "opacity": 0.5,
            "scalar_bar": True,  # mappé vers show_scalar_bar
            "name": t.get_entities().get('bundle'),
            "style": "surface",
        }
    )
#Add /home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz
vis.add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz",
        {
            "display_array": "intensity",
            "cmap": "gray",
            "clim": (200, 800),
            "opacity": 0.3,
            "scalar_bar": False,
            "name": "anatomy",
            "ambient": 0.6,
            "specular": 0.1,
            "diffuse": 0.8,
            "style": "surface",
        }
    )

vis.show()