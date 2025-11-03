import os 
import pandas as pd
import numpy as np
import pyvista
import json
from tractviewer import TractViewer

# Configuration des paramètres
target_var= 'group'
target_bundle = 'STPREFleft'
target_metric = ['IFW','FA']
if target_metric=='ALL' :
    target_metric=['FA','MD','RD','AD','IFW','IRF']

target_csv= "/home/ndecaux/report_optimized_no_actimetry_clusterFWE/summary_results.csv"

# Modes d'affichage disponibles
DISPLAY_MODES = {
    'r_values': 'Valeurs r complètes',
    'significance': 'Zones significatives (binaire)',
    'p_values': 'Valeurs p',
    'r_significant': 'Valeurs r sur zones significatives uniquement'
}

# Mode d'affichage choisi
DISPLAY_MODE = 'significance'  # Changez ici pour choisir le mode

base_dir = os.path.dirname(target_csv)

out_dir=f'/home/ndecaux/bundle_profiles/{base_dir.split("/")[-1]}_{target_bundle}_{target_var}'
os.makedirs(out_dir, exist_ok=True)

# Charger les données
df = pd.read_csv(target_csv)

# Filtrer et formater les données comme dans le notebook
MIN_SIG = 1
MAX_REMOVED_SUBJ = 1
MAX_REMOVED_POINTS = 3

filters = {
    "significant": df['n_sig_corrected'] > MIN_SIG-1,
    "all_subs": df['removed_subjects'] < MAX_REMOVED_SUBJ+1,
    "all_points": df['removed_points'] < MAX_REMOVED_POINTS+1,
}

final_filter = filters["significant"] & filters["all_subs"] & filters["all_points"]
df_filtered = df[final_filter]

# Séparer type et val
df_filtered['val'] = df_filtered['type'].apply(lambda x: '_'.join(x.split('_')[1:]))
df_filtered['type'] = df_filtered['type'].apply(lambda x: x.split('_')[0])

# Construire les chemins des fichiers
df_filtered['csv'] = df_filtered.apply(lambda row: os.path.join(
    base_dir, 'figures', 
    f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else 'partial'}.csv"
), axis=1)

df_filtered['png'] = df_filtered.apply(lambda row: os.path.join(
    base_dir, 'figures',
    f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else 'partial'}.png"
), axis=1)

# Filtrer pour le bundle et la variable cibles
target_row = df_filtered[
    (df_filtered['bundle'] == target_bundle) & 
    (df_filtered['val'] == target_var) &
    (df_filtered['metric'].isin(target_metric))
]

if len(target_row) == 0:
    print(f"Aucune donnée trouvée pour {target_bundle} et {target_var}")
    exit()

# Charger les descriptions des bundles
with open('/home/ndecaux/Code/actiDep/bundle_desc_fr.json', encoding='utf-8') as f:
    bundle_desc = json.load(f)
bundle_desc = {k.replace('_',''): v for k, v in bundle_desc.items()}

# Modèle de tractographie
model_template = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_50pts/sub-01002/tracto/sub-01002_bundle-<BUNDLE>_desc-associations_model-MCM_space-HCP_tracto.vtk"

results_summary = []

for idx, row in target_row.iterrows():
    bundle = row['bundle']
    metric = row['metric']
    viewer = TractViewer()

    # Ajouter l'anatomie
    viewer.add_dataset(
        "/home/ndecaux/Code/Data/Atlas/atlas_anat.nii.gz",
        {
            "display_array": "intensity",
            "cmap": "gray",
            "clim": (200, 800),
            "opacity": 0.1,
            "ambient": 0.6,
            "diffuse": 0.8,
            "specular": 0.1,
            "scalar_bar": False,
            "name": "anatomy",
            "style": "surface",
        }
    )
    print(f'Processing {metric} {bundle} with mode {DISPLAY_MODE}')
    
    # Charger le modèle de tractographie
    model_path = model_template.replace('<BUNDLE>', bundle)
    if not os.path.exists(model_path):
        print(f"Modèle non trouvé: {model_path}")
        continue
        
    mesh = pyvista.read(model_path)
    
    # Charger les données statistiques
    if not os.path.exists(row['csv']):
        print(f"CSV non trouvé: {row['csv']}")
        continue
        
    csv_data = pd.read_csv(row['csv'])
    
    # Analyser les clusters significatifs
    clusters = []
    current_cluster = []
    for i, val in enumerate(csv_data['sig_afq']):
        if val:
            current_cluster.append(i)
        else:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
    if current_cluster:
        clusters.append(current_cluster)
    
    cluster_info = []
    for c_idx, c in enumerate(clusters):
        mean_p_raw = csv_data.loc[c, 'p_raw'].mean()
        n_points = len(c)
        
        if 'r' in csv_data.columns:
            r = csv_data.loc[c, 'r'].mean()
            text = f'Cluster {c_idx+1} : n={n_points} pts ; p={mean_p_raw:.3f} ; r={r:.3f}'
            cluster_info.append({
                'start': c[0], 'end': c[-1], 'n_points': n_points, 
                'mean_p_raw': mean_p_raw, 'r': r, 'text': text
            })
    
    # Filtrer les données significatives
    csv_sig = csv_data[csv_data['sig_afq']]
    if len(csv_sig) == 0:
        print(f"Aucun point significatif pour {bundle} {metric}")
        continue
    
    # Préparer les données selon le mode d'affichage
    if DISPLAY_MODE == 'r_values':
        # Afficher toutes les valeurs r
        if 'r' in csv_data.columns:
            p2r = dict(zip(csv_data['point'].astype(int), csv_data['r'].astype(float)))
            value_col = 'r'
            cmap = "coolwarm"
        else:
            p2r = dict(zip(csv_data['point'].astype(int), csv_data['diff'].astype(float)))
            value_col = 'diff'
            cmap = "coolwarm"
            
    elif DISPLAY_MODE == 'significance':
        # Afficher zones significatives en binaire (1=significatif, 0=non)
        p2r = dict(zip(csv_data['point'].astype(int), csv_data['sig_afq'].astype(float)))
        value_col = 'significance'
        cmap = "Reds"
        
    elif DISPLAY_MODE == 'p_values':
        # Afficher les valeurs p (inversées pour que plus rouge = plus significatif)
        p_inv = 1 - csv_data['p_raw']  # Inverser pour avoir rouge = significatif
        p2r = dict(zip(csv_data['point'].astype(int), p_inv.astype(float)))
        value_col = 'p_inverse'
        cmap = "coolwarm"
        
    elif DISPLAY_MODE == 'r_significant':
        # Afficher valeurs r uniquement sur zones significatives
        csv_sig = csv_data[csv_data['sig_afq']]
        if 'r' in csv_sig.columns:
            p2r = dict(zip(csv_sig['point'].astype(int), csv_sig['r'].astype(float)))
            value_col = 'r_sig'
        else:
            p2r = dict(zip(csv_sig['point'].astype(int), csv_sig['diff'].astype(float)))
            value_col = 'diff_sig'
        # cmap = "coolwarm"
        cmap = "cold_hot"    
    # Vérifier si on a des données à afficher
    if len(p2r) == 0:
        print(f"Aucune donnée à afficher pour {bundle} {metric} en mode {DISPLAY_MODE}")
        continue
    
    pi = np.asarray(mesh.point_data['point_index']).astype(int)
    values = np.array([p2r.get(i, 0.0) for i in pi], dtype=float)
    
    # Nommer l'array selon le mode d'affichage pour une légende parlante
    array_names = {
        'r_values': 'Correlation (r)',
        'significance': 'Significance',
        'p_values': '1 - p-value',
        'r_significant': 'r (significant only)'
    }
    
    array_name = array_names.get(DISPLAY_MODE, 'stats')
    mesh.point_data[array_name] = values
    
    # Sauvegarder le VTK
    output_vtk = os.path.join(out_dir, f"{bundle}_{metric}_{target_var}_{DISPLAY_MODE}.vtk")
    mesh.save(output_vtk)
    
    # Calculer les limites d'affichage selon le mode
    if DISPLAY_MODE == 'significance':
        clim = (0, 1)
    elif DISPLAY_MODE == 'p_values':
        clim = (0, 1)
    else:
        # Pour r_values et r_significant
        non_zero_values = values[values != 0]
        if len(non_zero_values) > 0:
            clim = (non_zero_values.min(), non_zero_values.max())
        else:
            clim = (-1, 1)
    
    # Ajouter au viewer
    viewer.add_dataset(
        output_vtk,
        {
            "display_array": array_name,
            "cmap": cmap,
            "clim": clim,
            "opacity": 0.8,
            "ambient": 0.3,
            "diffuse": 0.6,
            "scalar_bar": True,
            "name": f"{bundle}_{metric}_{DISPLAY_MODE}"
        }
    )
    
    # Copier les fichiers PNG
    if os.path.exists(row['png']):
        output_png = os.path.join(out_dir, f"{bundle}_{metric}_{target_var}.png")
        os.system(f'cp "{row["png"]}" "{output_png}"')
    
    results_summary.append({
        'bundle': bundle,
        'metric': metric,
        'clusters': cluster_info,
        'vtk': output_vtk,
        'bundle_desc': bundle_desc.get(bundle, bundle),
        'display_mode': DISPLAY_MODE,
        'display_desc': DISPLAY_MODES[DISPLAY_MODE]
    })

    # Générer l'animation
    output_gif = os.path.join(out_dir, f'{target_bundle}_{metric}_{target_var}_{DISPLAY_MODE}_visualization.gif') 
    viewer.record_rotation(
        output_gif,
        n_frames=180,
        step=2,
        elevation=0.0,
        fps=10,
        quality=9,
        crf=18,
        supersample=2,
        window_size=(600, 400),
        rotation_x= -90
    )

    viewer.capture_screenshot(os.path.join(out_dir, f'{bundle}_{metric}_{target_var}_{DISPLAY_MODE}_screenshot.png'),rotation_x= -90,rotation_y=-45)

    viewer

# Créer le résumé
summary_text = f'Profil du bundle {target_bundle} pour la variable {target_var}\n'
summary_text += f'Mode d\'affichage : {DISPLAY_MODES[DISPLAY_MODE]}\n\n'

for result in results_summary:
    summary_text += f'Bundle {result["bundle_desc"]} ({result["bundle"]}) - Métrique {result["metric"]}:\n'
    for cluster in result['clusters']:
        summary_text += f"  - {cluster['text']}\n"
    summary_text += '\n'

summary_text += f'\nModes d\'affichage disponibles :\n'
for mode, desc in DISPLAY_MODES.items():
    marker = " -> " if mode == DISPLAY_MODE else "    "
    summary_text += f'{marker}{mode}: {desc}\n'

summary_text += f'\nAnimation : {output_gif}\n'

with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"Visualisation générée dans : {out_dir}")
print(f"Mode d'affichage : {DISPLAY_MODES[DISPLAY_MODE]}")
print(f"Animation sauvegardée : {output_gif}")


