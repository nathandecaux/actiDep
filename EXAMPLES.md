# Exemples d'Utilisation - actiDep

Ce fichier contient des exemples pratiques d'utilisation du package `actiDep` pour différents cas d'usage courants dans l'analyse de données de diffusion.

## 📋 Table des Matières

- [Configuration Initiale](#configuration-initiale)
- [Chargement des Données](#chargement-des-données)
- [Analyse Tractométrique](#analyse-tractométrique)
- [Pipeline Complet](#pipeline-complet)
- [Visualisation](#visualisation)
- [Analyse Avancée](#analyse-avancée)

## ⚙️ Configuration Initiale

### Configuration des Outils Externes

```python
from actiDep.set_config import set_config

# Configuration automatique des outils
config, tools = set_config()

print("Configuration chargée :")
print(f"ANIMA: {config['animaDir']}")
print(f"MRtrix: {tools['mrtrix']}")
print(f"TractSeg: {tools['tractseg']}")
```

### Vérification de l'Installation

```python
import actiDep
from actiDep.data.loader import Actidep
import numpy as np
import nibabel as nib

print(f"Version actiDep: {actiDep.__version__}")
print("Installation vérifiée avec succès !")
```

## 📂 Chargement des Données

### Chargement d'un Dataset BIDS

```python
from actiDep.data.loader import Actidep, Subject

# Initialisation du dataset
dataset_path = "/path/to/actidep/bids"
dataset = Actidep(dataset_path)

# Lister tous les sujets
subjects = dataset.get_subjects()
print(f"Nombre de sujets: {len(subjects)}")

# Informations sur le dataset
print(f"Pipelines disponibles: {dataset.get_pipelines()}")
print(f"Datatypes disponibles: {dataset.get_datatypes()}")
```

### Manipulation d'un Sujet Individuel

```python
# Chargement d'un sujet spécifique
subject = Subject("01002", db_root=dataset_path)

# Récupération de fichiers par critères
dwi_files = subject.get(
    scope='raw',
    suffix='dwi',
    extension='nii.gz'
)

fa_files = subject.get(
    pipeline='anima_preproc',
    metric='FA',
    extension='nii.gz'
)

tractograms = subject.get(
    pipeline='bundle_seg',
    suffix='tracto',
    extension='trk'
)

print(f"Fichiers DWI: {len(dwi_files)}")
print(f"Cartes FA: {len(fa_files)}")
print(f"Tractogrammes: {len(tractograms)}")
```

### Gestion des Métadonnées

```python
# Accès aux entités d'un fichier
fa_file = fa_files[0]
print(f"Chemin: {fa_file.path}")
print(f"Entités: {fa_file.get_entities()}")
print(f"Pipeline: {fa_file.pipeline}")
print(f"Sujet: {fa_file.subject}")

# Filtrage avancé
bundles = subject.get(
    pipeline='bundle_seg',
    bundle=['CSTleft', 'CSTright'],
    suffix='tracto'
)
```

## 📊 Analyse Tractométrique

### Analyse Basique avec TractSeg

```python
from actiDep.analysis.tractometry import process_tractseg_analysis

# Fichier contenant la liste des sujets
subjects_file = "subjects.txt"

# Analyse tractométrique pour la FA
results_fa = process_tractseg_analysis(
    subjects_txt=subjects_file,
    dataset_path=dataset_path,
    metric='FA',
    with_3dplot=False
)

# Analyse pour différentes métriques
for metric in ['FA', 'MD', 'RD', 'AD']:
    print(f"Traitement métrique: {metric}")
    process_tractseg_analysis(
        subjects_txt=subjects_file,
        dataset_path=dataset_path,
        metric=metric,
        with_3dplot=False
    )
```

### Analyse Tractométrique Personnalisée

```python
from actiDep.analysis.tractometry import evaluate_along_streamlines, process_projection
import nibabel as nib
from dipy.io.streamline import load_trk

# Chargement manuel des données
fa_file = subject.get_unique(pipeline='anima_preproc', metric='FA')
tracto_file = subject.get_unique(pipeline='bundle_seg', bundle='CSTleft', suffix='tracto')

# Chargement des images et tractogrammes
fa_img = nib.load(fa_file.path)
fa_data = fa_img.get_fdata()
affine = fa_img.affine

tractogram = load_trk(tracto_file.path, 'same', bbox_valid_check=False)
streamlines = tractogram.streamlines

# Évaluation le long des streamlines
means, stds = evaluate_along_streamlines(
    scalar_img=fa_data,
    streamlines=streamlines,
    nr_points=100,
    affine=affine,
    algorithm='distance_map'  # ou 'equal_dist', 'cutting_plane', 'afq'
)

print(f"Profil FA - Moyenne: {np.mean(means):.3f}")
print(f"Profil FA - Écart-type: {np.mean(stds):.3f}")

# Visualisation du profil
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(means, label='Moyenne FA', linewidth=2)
plt.fill_between(range(len(means)), 
                 np.array(means) - np.array(stds),
                 np.array(means) + np.array(stds),
                 alpha=0.3, label='±1 écart-type')
plt.xlabel('Position le long du faisceau')
plt.ylabel('Anisotropie Fractionnelle')
plt.title(f'Profil tractométrique - {tracto_file.bundle}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'profile_{tracto_file.bundle}_FA.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Analyse Multi-Métriques

```python
# Analyse pour plusieurs métriques et faisceaux
metrics = ['FA', 'MD', 'RD', 'AD']
bundles = ['CSTleft', 'CSTright', 'UFright', 'UFleft']

results = {}

for bundle in bundles:
    results[bundle] = {}
    tracto = subject.get_unique(
        pipeline='bundle_seg', 
        bundle=bundle, 
        suffix='tracto'
    )
    
    for metric in metrics:
        try:
            metric_file = subject.get_unique(
                pipeline='anima_preproc',
                metric=metric
            )
            
            # Chargement et analyse
            metric_img = nib.load(metric_file.path)
            metric_data = metric_img.get_fdata()
            
            tractogram = load_trk(tracto.path, 'same', bbox_valid_check=False)
            
            means, stds = evaluate_along_streamlines(
                scalar_img=metric_data,
                streamlines=tractogram.streamlines,
                nr_points=100,
                affine=metric_img.affine
            )
            
            results[bundle][metric] = {
                'means': means,
                'stds': stds,
                'global_mean': np.mean(means),
                'global_std': np.mean(stds)
            }
            
            print(f"{bundle} - {metric}: {np.mean(means):.3f} ± {np.mean(stds):.3f}")
            
        except Exception as e:
            print(f"Erreur pour {bundle}-{metric}: {e}")
            continue

# Sauvegarde des résultats
import json
with open(f'tractometry_results_{subject.sub_id}.json', 'w') as f:
    # Conversion des arrays numpy en listes pour JSON
    json_results = {}
    for bundle, bundle_data in results.items():
        json_results[bundle] = {}
        for metric, metric_data in bundle_data.items():
            json_results[bundle][metric] = {
                'means': [float(x) for x in metric_data['means']],
                'stds': [float(x) for x in metric_data['stds']],
                'global_mean': float(metric_data['global_mean']),
                'global_std': float(metric_data['global_std'])
            }
    json.dump(json_results, f, indent=2)
```

## 🔄 Pipeline Complet

### Pipeline Multi-Shell Multi-Tissue CSD

```python
from actiDep.pipeline.msmt_csd import (
    process_response, process_fod, process_normalize,
    process_fixels, process_ifod2_tracto
)

# Configuration du pipeline
pipeline_name = "msmt_csd"

# 1. Calcul des réponses tissulaires
print("Étape 1: Calcul des réponses tissulaires")
dwi_file = subject.get_unique(scope='raw', suffix='dwi')
bval_file = subject.get_unique(scope='raw', suffix='dwi', extension='bval')
bvec_file = subject.get_unique(scope='raw', suffix='dwi', extension='bvec')

responses = process_response(
    subject=subject,
    dwi_data={'dwi': dwi_file, 'bval': bval_file, 'bvec': bvec_file},
    pipeline=pipeline_name
)

# 2. Estimation des FOD
print("Étape 2: Estimation des FOD")
fods = process_fod(
    subject=subject,
    dwi_data={'dwi': dwi_file, 'bval': bval_file, 'bvec': bvec_file},
    pipeline=pipeline_name
)

# 3. Normalisation des FOD
print("Étape 3: Normalisation")
normalized_fods = process_normalize(
    subject=subject,
    pipeline=pipeline_name
)

# 4. Analyse basée sur les fixels
print("Étape 4: Analyse fixels")
fixels = process_fixels(
    subject=subject,
    pipeline=pipeline_name
)

# 5. Tractographie iFOD2
print("Étape 5: Tractographie")
tractogram = process_ifod2_tracto(
    subject=subject,
    pipeline=pipeline_name
)

print("Pipeline MSMT-CSD terminé avec succès !")
```

### Pipeline de Segmentation de Faisceaux

```python
from actiDep.pipeline.bundle_seg import run_bundle_segmentation
from actiDep.utils.recobundle import create_whole_brain_tract

# Segmentation automatique
print("Segmentation des faisceaux...")
segmentation_results = run_bundle_segmentation(
    subject=subject,
    pipeline="bundle_seg"
)

# Vérification des résultats
segmented_bundles = subject.get(
    pipeline='bundle_seg',
    suffix='tracto',
    extension='trk'
)

print(f"Faisceaux segmentés: {len(segmented_bundles)}")
for bundle in segmented_bundles[:5]:  # Premiers 5 faisceaux
    print(f"- {bundle.bundle}: {bundle.path}")

# Création d'un tractogramme cerveau entier
print("Création du tractogramme cerveau entier...")
whole_brain = create_whole_brain_tract(
    tract_list=[b.path for b in segmented_bundles],
    ref_image=dwi_file.path
)

print("Segmentation terminée !")
```

## 🎨 Visualisation

### Interface Interactive de Clustering

```python
from actiDep.visualisation.centroids_params import TractoClusteringApp

# Lancement de l'interface interactive
app = TractoClusteringApp()

# Chargement des données
tractogram_file = segmented_bundles[0].path  # Premier faisceau
anatomy_file = fa_files[0].path

app.load_data(
    tractogram_file=tractogram_file,
    anatomy_file=anatomy_file
)

# Interface graphique pour ajuster les paramètres de clustering
print("Interface de clustering lancée...")
print("Utilisez les contrôles pour ajuster les paramètres de QuickBundles")
```

### Visualisation avec DIPY

```python
from dipy.viz import window, actor
from dipy.io.streamline import load_trk

# Préparation des données
scene = window.Scene()
scene.SetBackground(1, 1, 1)  # Fond blanc

# Chargement d'un faisceau
bundle_file = subject.get_unique(
    pipeline='bundle_seg',
    bundle='CSTleft',
    suffix='tracto'
)

tractogram = load_trk(bundle_file.path, 'same', bbox_valid_check=False)
streamlines = tractogram.streamlines

# Ajout des streamlines à la scène
streamlines_actor = actor.line(streamlines, colors=(1, 0, 0))  # Rouge
scene.add(streamlines_actor)

# Ajout d'une carte FA en arrière-plan (optionnel)
fa_file = subject.get_unique(pipeline='anima_preproc', metric='FA')
fa_img = nib.load(fa_file.path)
fa_data = fa_img.get_fdata()

# Coupe axiale
slice_actor = actor.slicer(fa_data, affine=fa_img.affine)
slice_actor.display(z=fa_data.shape[2]//2)
scene.add(slice_actor)

# Sauvegarde de l'image
window.record(scene, out_path=f'visualization_{bundle_file.bundle}.png', size=(800, 600))
print(f"Visualisation sauvegardée: visualization_{bundle_file.bundle}.png")
```

### Création de GIFs Rotatifs

```python
from actiDep.visualisation.create_rotating_3D_gif import create_rotating_gif

# Création d'un GIF rotatif
create_rotating_gif(
    tractogram_path=bundle_file.path,
    reference_path=fa_file.path,
    output_path=f'rotating_{bundle_file.bundle}.gif',
    n_frames=36,  # 36 frames pour rotation complète
    elevation=10,
    azimuth_step=10
)

print(f"GIF rotatif créé: rotating_{bundle_file.bundle}.gif")
```

## 🔬 Analyse Avancée

### Analyse Microstructurelle avec MCM

```python
from actiDep.analysis.microstructure import MCMVTKReader
from actiDep.utils.mcm import project_to_central_line_from_vtk

# Chargement d'un tractogramme MCM
mcm_file = subject.get_unique(
    pipeline='mcm_tensors',
    suffix='tracto',
    extension='vtk'
)

# Lecture des données MCM
reader = MCMVTKReader(mcm_file.path)

# Affichage des métadonnées
metadata = reader.get_metadata()
print("Métadonnées MCM:")
for key, value in metadata.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key}: {value[:3]}... ({len(value)} éléments)")
    else:
        print(f"  {key}: {value}")

# Extraction de streamlines avec paramètres MCM
print("\nExtraction de 10 streamlines...")
streamlines_mcm = reader.extract_streamlines(0, 10)

for i, streamline in enumerate(streamlines_mcm):
    print(f"Streamline {i}: {len(streamline['points'])} points")
    if 'fa' in streamline:
        print(f"  FA moyenne: {np.mean(streamline['fa']):.3f}")

# Projection sur ligne centrale
print("Projection sur ligne centrale...")
reference_file = subject.get_unique(pipeline='anima_preproc', metric='FA')

project_to_central_line_from_vtk(
    vtk_file_path=mcm_file.path,
    reference_nifti_path=reference_file.path,
    output_path=f'central_line_{subject.sub_id}.vtk',
    num_points_central_line=100
)

print("Projection terminée !")
```

### Analyse de Densité de Fibres

```python
from actiDep.analysis.fiber_density import average_with_ants

# Création d'un template de densité de fibres
print("Création du template de densité de fibres...")

# Utilisation de tous les sujets du dataset
average_with_ants(
    dataset=dataset,
    output_dir="/path/to/output/fiber_density_atlas"
)

print("Template créé avec succès !")
```

### Clustering Avancé de Streamlines

```python
from actiDep.utils.clustering import cluster_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric

# Clustering d'un tractogramme
bundle_file = subject.get_unique(
    pipeline='bundle_seg',
    bundle='CSTleft',
    suffix='tracto'
)

# Clustering avec différents seuils
thresholds = [50, 100, 150, 200]
results = {}

for threshold in thresholds:
    print(f"Clustering avec seuil {threshold}mm...")
    
    clusters = cluster_streamlines(
        tractogram=bundle_file.path,
        threshold=threshold
    )
    
    results[threshold] = {
        'n_clusters': len(clusters),
        'cluster_sizes': [len(cluster) for cluster in clusters]
    }
    
    print(f"  Nombre de clusters: {len(clusters)}")
    print(f"  Taille moyenne: {np.mean([len(c) for c in clusters]):.1f}")

# Analyse des résultats
import pandas as pd

df_results = pd.DataFrame([
    {
        'threshold': t,
        'n_clusters': r['n_clusters'],
        'avg_size': np.mean(r['cluster_sizes']),
        'std_size': np.std(r['cluster_sizes'])
    }
    for t, r in results.items()
])

print("\nRésultats du clustering:")
print(df_results)

# Sauvegarde
df_results.to_csv(f'clustering_analysis_{bundle_file.bundle}.csv', index=False)
```

### Analyse Statistique Multi-Sujets

```python
import pandas as pd
from scipy import stats

# Collecte des données pour tous les sujets
all_subjects = dataset.get_subjects()
bundle_name = 'CSTleft'
metric = 'FA'

group_data = []

for subj_id in all_subjects[:10]:  # Premiers 10 sujets
    try:
        subj = Subject(subj_id, db_root=dataset_path)
        
        # Récupération des données tractométriques
        tracto_file = subj.get_unique(
            pipeline='tractometry',
            bundle=bundle_name,
            suffix='mean',
            metric=metric,
            extension='csv'
        )
        
        # Chargement des données
        data = pd.read_csv(tracto_file.path, sep=';')
        
        # Extraction des valeurs moyennes
        fa_values = data[bundle_name].values
        
        group_data.append({
            'subject': subj_id,
            'bundle': bundle_name,
            'metric': metric,
            'values': fa_values,
            'mean_value': np.mean(fa_values),
            'std_value': np.std(fa_values)
        })
        
    except Exception as e:
        print(f"Erreur pour sujet {subj_id}: {e}")
        continue

# Création du DataFrame
df_group = pd.DataFrame(group_data)

# Statistiques descriptives
print(f"Analyse de groupe - {bundle_name} ({metric}):")
print(f"Nombre de sujets: {len(df_group)}")
print(f"Moyenne globale: {df_group['mean_value'].mean():.3f} ± {df_group['mean_value'].std():.3f}")
print(f"Min: {df_group['mean_value'].min():.3f}")
print(f"Max: {df_group['mean_value'].max():.3f}")

# Test de normalité
statistic, p_value = stats.shapiro(df_group['mean_value'])
print(f"Test de Shapiro-Wilk: p = {p_value:.3f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_group['mean_value'], bins=10, alpha=0.7, edgecolor='black')
plt.xlabel(f'{metric} moyenne')
plt.ylabel('Fréquence')
plt.title(f'Distribution {metric} - {bundle_name}')

plt.subplot(1, 2, 2)
for i, row in df_group.iterrows():
    plt.plot(row['values'], alpha=0.3, color='blue')

# Calcul et affichage de la moyenne de groupe
mean_profile = np.mean([row['values'] for _, row in df_group.iterrows()], axis=0)
std_profile = np.std([row['values'] for _, row in df_group.iterrows()], axis=0)

plt.plot(mean_profile, 'r-', linewidth=2, label='Moyenne groupe')
plt.fill_between(range(len(mean_profile)),
                 mean_profile - std_profile,
                 mean_profile + std_profile,
                 alpha=0.3, color='red', label='±1 écart-type')

plt.xlabel('Position le long du faisceau')
plt.ylabel(metric)
plt.title(f'Profils tractométriques - {bundle_name}')
plt.legend()

plt.tight_layout()
plt.savefig(f'group_analysis_{bundle_name}_{metric}.png', dpi=300, bbox_inches='tight')
plt.show()

# Sauvegarde des résultats
df_group.to_csv(f'group_analysis_{bundle_name}_{metric}.csv', index=False)
print(f"Analyse sauvegardée: group_analysis_{bundle_name}_{metric}.csv")
```

## 🔧 Troubleshooting

### Problèmes Courants et Solutions

```python
# Vérification de l'intégrité des données
def check_data_integrity(subject, verbose=True):
    """Vérifie l'intégrité des données d'un sujet."""
    
    issues = []
    
    # Vérification DWI
    try:
        dwi_files = subject.get(scope='raw', suffix='dwi')
        if not dwi_files:
            issues.append("Aucun fichier DWI trouvé")
        elif verbose:
            print(f"✓ DWI: {len(dwi_files)} fichier(s)")
    except Exception as e:
        issues.append(f"Erreur DWI: {e}")
    
    # Vérification prétraitement
    try:
        fa_files = subject.get(pipeline='anima_preproc', metric='FA')
        if not fa_files:
            issues.append("Pas de carte FA (prétraitement requis)")
        elif verbose:
            print(f"✓ FA: {len(fa_files)} fichier(s)")
    except Exception as e:
        issues.append(f"Erreur FA: {e}")
    
    # Vérification faisceaux
    try:
        bundles = subject.get(pipeline='bundle_seg', suffix='tracto')
        if not bundles:
            issues.append("Pas de faisceaux segmentés")
        elif verbose:
            print(f"✓ Faisceaux: {len(bundles)} fichier(s)")
    except Exception as e:
        issues.append(f"Erreur faisceaux: {e}")
    
    if issues:
        print(f"⚠️  Problèmes détectés pour {subject.sub_id}:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"✅ Données complètes pour {subject.sub_id}")
    
    return len(issues) == 0

# Vérification pour tous les sujets
print("Vérification de l'intégrité des données...")
for subj_id in dataset.get_subjects()[:5]:
    subj = Subject(subj_id, db_root=dataset_path)
    check_data_integrity(subj, verbose=False)
```

Ces exemples couvrent les principales fonctionnalités d'`actiDep` et peuvent servir de base pour développer des analyses plus spécialisées selon vos besoins spécifiques.
