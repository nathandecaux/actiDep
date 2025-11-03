#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour créer une figure présentant les bundles 2D avec leurs
représentations 3D correspondantes au-dessus.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import re
import sys

# Configuration - mettre à True pour filtrer selon subjects.txt
FILTER_BY_SUBJECTS_FILE = True  # Changer à False pour traiter tous les bundles

def load_subjects_from_file(subjects_file_path):
    """
    Charge la liste des sujets depuis le fichier subjects.txt
    Retourne un set des subject_ids
    """
    subjects = set()
    try:
        with open(subjects_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignorer les commentaires et les lignes vides
                if line and not line.startswith('#') and not line.startswith('subject_id'):
                    parts = line.split()
                    if len(parts) >= 1:
                        subjects.add(parts[0])  # Premier élément est le subject_id
        print(f"Sujets trouvés dans {subjects_file_path}: {sorted(subjects)}")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {subjects_file_path}: {e}", file=sys.stderr)
        return set()
    return subjects

def filter_bundles_by_subjects(bundles, valid_subjects):
    """
    Filtre les bundles pour ne garder que ceux correspondant aux sujets valides
    """
    filtered_bundles = []
    for bundle in bundles:
        # Extraire le subject_id du nom du bundle
        # Assumer que le format est quelque chose comme "sub-XXXXX_bundle_name"
        parts = bundle.split('_')
        if len(parts) > 0:
            potential_subject = parts[0]
            if potential_subject in valid_subjects:
                filtered_bundles.append(bundle)
    return filtered_bundles

# Dossier contenant les images
input_dir = "/home/ndecaux/Code/actiDep/ist/FW_with3d/good_subplots"
output_dir = "/home/ndecaux/Code/actiDep/output_subplots"
subjects_file = "/home/ndecaux/Code/actiDep/subjects.txt"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Charger la liste des sujets valides si le filtrage est activé
valid_subjects = set()
if FILTER_BY_SUBJECTS_FILE:
    valid_subjects = load_subjects_from_file(subjects_file)
    if not valid_subjects:
        print("Aucun sujet trouvé dans le fichier subjects.txt, traitement de tous les bundles.")
        FILTER_BY_SUBJECTS_FILE = False

# Obtenir la liste des fichiers dans le dossier
try:
    files = os.listdir(input_dir)
    print(f"Fichiers trouvés dans {input_dir}: {files}")
except Exception as e:
    print(f"Erreur lors de la lecture du dossier {input_dir}: {e}", file=sys.stderr)
    sys.exit(1)

# Filtrer les fichiers pour obtenir les images 2D et 3D
bundles_2d = [f for f in files if not f.startswith("tractometry_results_")]
bundles_3d = [f for f in files if f.startswith("tractometry_results_")]

# Filtrer par sujets si activé
if FILTER_BY_SUBJECTS_FILE and valid_subjects:
    bundles_2d = filter_bundles_by_subjects(bundles_2d, valid_subjects)
    bundles_3d = filter_bundles_by_subjects(bundles_3d, valid_subjects)

print(f"Images 2D trouvées: {bundles_2d}")
print(f"Images 3D trouvées: {bundles_3d}")

# Créer un dictionnaire pour associer chaque bundle à ses images 2D et 3D
bundle_dict = {}
for img_2d in bundles_2d:
    bundle_name = os.path.splitext(img_2d)[0]
    img_3d_name = f"tractometry_results_{bundle_name}_3D.png"
    
    if img_3d_name in bundles_3d:
        bundle_dict[bundle_name] = {
            "2d": img_2d,
            "3d": img_3d_name
        }

# Trier les bundles par nom pour un affichage cohérent
sorted_bundles = sorted(bundle_dict.keys())
num_bundles = len(sorted_bundles)

# Créer la figure
# Augmenter la hauteur de la figure pour laisser de l'espace à la légende en bas
fig = plt.figure(figsize=(num_bundles * 4, 9))  # Hauteur augmentée à 9 (au lieu de 8)

# Calculer la largeur d'une sous-figure
subfig_width = 1.0 / num_bundles

# Définir les proportions verticales initiales pour les images
orig_height_3d = 0.36
orig_height_2d = 0.56

legend_fig_height_fraction = 0.07  # Fraction de la hauteur de la figure pour la légende
images_total_fig_height_fraction = 1.0 - legend_fig_height_fraction

# Calculer les proportions des images 2D et 3D à l'intérieur de leur bloc combiné
# Ces proportions internes sommeront à 1.0
total_initial_prop_sum = orig_height_3d + orig_height_2d
if total_initial_prop_sum <= 1e-6:  # Éviter la division par zéro
    prop_2d_of_img_block = 0.5
    prop_3d_of_img_block = 0.5
else:
    prop_2d_of_img_block = orig_height_2d / total_initial_prop_sum
    prop_3d_of_img_block = orig_height_3d / total_initial_prop_sum

# Calculer les hauteurs réelles des images en coordonnées de figure
actual_fig_h_2d = prop_2d_of_img_block * images_total_fig_height_fraction
actual_fig_h_3d = prop_3d_of_img_block * images_total_fig_height_fraction

# Calculer les positions Y de base pour les blocs d'images 2D et 3D
# Les images 2D commencent juste au-dessus de la légende
y_pos_2d_bottom = legend_fig_height_fraction
# Les images 3D commencent juste au-dessus des images 2D
y_pos_3d_bottom = legend_fig_height_fraction + actual_fig_h_2d

# Ajouter les images à la figure
for i, bundle in enumerate(sorted_bundles):
    # Position de base pour cette colonne
    x_pos = i * subfig_width
    
    # Image 3D en haut
    ax_3d = fig.add_axes([x_pos, y_pos_3d_bottom, subfig_width, actual_fig_h_3d])
    img_3d = imread(os.path.join(input_dir, bundle_dict[bundle]["3d"]))
    
    ax_3d.imshow(img_3d)
    ax_3d.set_title(bundle, pad=0)
    ax_3d.axis('off')
    
    # Image 2D en bas
    ax_2d = fig.add_axes([x_pos, y_pos_2d_bottom, subfig_width, actual_fig_h_2d])
    img_2d = imread(os.path.join(input_dir, bundle_dict[bundle]["2d"]))
    ax_2d.imshow(img_2d)
    ax_2d.axis('off')

# Ajouter une légende commune en bas de la figure
# Créer un axe spécifique pour la légende avec une taille augmentée
# L'axe de la légende occupe toute la largeur et la hauteur réservée en bas.
legend_ax = fig.add_axes([0, 0, 1.0, legend_fig_height_fraction])

# Créer les éléments de la légende (en anglais)
# Utiliser les couleurs du cycle de couleurs par défaut de matplotlib
blue_patch = mpatches.Patch(color='#1f77b4', label='Controls')  # Bleu - première couleur du cycle par défaut
orange_patch = mpatches.Patch(color='#ff7f0e', label='Patients')  # Orange - deuxième couleur du cycle par défaut
red_line = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Significant results')

# Ajouter la légende avec une taille de police augmentée
legend = legend_ax.legend(handles=[blue_patch, orange_patch, red_line], loc='center', ncol=3, frameon=False, fontsize=14)
legend_ax.axis('off')  # Masquer les axes de la légende

# Ajuster les marges de la figure pour qu'elles soient complètement nulles
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

# Enregistrer la figure avec la légende
output_path = os.path.join(output_dir, "bundle_comparison_figure.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Utiliser bbox_inches='tight' pour inclure la légende
print(f"Figure enregistrée: {output_path}")

# Afficher la figure
plt.close()  # Fermer la figure au lieu de l'afficher pour éviter les problèmes d'affichage en mode non-interactif
