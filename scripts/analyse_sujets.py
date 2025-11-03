#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplifié pour analyser les données démographiques des sujets
"""

import pandas as pd
import numpy as np
from scipy import stats

# Lire les lignes du fichier
with open('/home/ndecaux/Code/actiDep/scripts/subjects.txt', 'r') as f:
    lines = f.readlines()

# Trouver la ligne d'en-tête
header_line = 0
for i, line in enumerate(lines):
    if line.startswith('subject_id'):
        header_line = i
        break

# Extraire les données
data = []
for line in lines[header_line+1:]:
    if line.strip():  # Ignorer les lignes vides
        parts = line.strip().split()
        subject_id = parts[0]
        group = int(parts[1])
        age = parts[2] if parts[2] != 'NA' else np.nan
        sex = parts[3] if parts[3] != 'NA' else np.nan
        
        if age != np.nan:
            try:
                age = float(age)
            except:
                age = np.nan
                
        if sex != np.nan:
            try:
                sex = float(sex)
            except:
                sex = np.nan
                
        data.append([subject_id, group, age, sex])

# Créer le DataFrame
df = pd.DataFrame(data, columns=['subject_id', 'group', 'age', 'sex'])

# Convertir les types
df['group'] = df['group'].astype(int)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['sex'] = pd.to_numeric(df['sex'], errors='coerce')

# Filtrer les données valides
df_valid = df.dropna(subset=['age', 'sex'])

# Conversion du sexe: 0 = hommes, 1 = femmes
df_valid['sex_label'] = df_valid['sex'].map({0.0: 'H', 1.0: 'F'})

# Groupes: 0 = témoins sains, 1 = patients dépressifs
group_labels = {0: 'Témoins', 1: 'Patients'}
df_valid['group_label'] = df_valid['group'].map(group_labels)

# Statistiques générales
total_subjects = len(df)
valid_subjects = len(df_valid)
print(f"Nombre total de sujets: {total_subjects}")
print(f"Sujets avec données complètes: {valid_subjects}")
print(f"Sujets avec données manquantes: {total_subjects - valid_subjects}")
print("-" * 50)

# Statistiques par groupe
for group_id in sorted(df_valid['group'].unique()):
    group_label = group_labels[group_id]
    group_data = df_valid[df_valid['group'] == group_id]
    n_subjects = len(group_data)
    age_mean = group_data['age'].mean()
    age_std = group_data['age'].std()
    n_female = sum(group_data['sex'] == 1.0)
    pct_female = (n_female / n_subjects) * 100
    
    print(f"{group_label} (groupe {group_id}):")
    print(f"  Nombre de sujets: {n_subjects}")
    print(f"  Âge: {age_mean:.1f} ± {age_std:.1f} ans (min: {group_data['age'].min():.1f}, max: {group_data['age'].max():.1f})")
    print(f"  Sexe: {n_female} femmes ({pct_female:.1f}%), {n_subjects - n_female} hommes ({100 - pct_female:.1f}%)")
    print()

# Test statistique pour comparer l'âge entre les groupes
group0 = df_valid[df_valid['group'] == 0]['age'].values
group1 = df_valid[df_valid['group'] == 1]['age'].values
t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)

print("COMPARAISON STATISTIQUE ENTRE GROUPES:")
print(f"Test t pour l'âge: t = {t_stat:.3f}, p = {p_val:.3f}")

# Créer une table de contingence pour le test du chi2
contingency_table = pd.crosstab(df_valid['group'], df_valid['sex'])
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency_table)
print(f"Test chi² pour la distribution des sexes: chi² = {chi2:.3f}, p = {p_chi2:.3f}")

# Résumé global tous groupes confondus
print("\nRÉSUMÉ GLOBAL (tous sujets confondus):")
all_age_mean = df_valid['age'].mean()
all_age_std = df_valid['age'].std()
all_n_female = sum(df_valid['sex'] == 1.0)
all_pct_female = (all_n_female / len(df_valid)) * 100
print(f"  Âge: {all_age_mean:.1f} ± {all_age_std:.1f} ans (min: {df_valid['age'].min():.1f}, max: {df_valid['age'].max():.1f})")
print(f"  Sexe: {all_n_female} femmes ({all_pct_female:.1f}%), {len(df_valid) - all_n_female} hommes ({100 - all_pct_female:.1f}%)")

# Compter le nombre de sujets par groupe
group_counts = df_valid['group'].value_counts().sort_index()
print(f"\nNombre de témoins (groupe 0): {group_counts[0]}")
print(f"Nombre de patients (groupe 1): {group_counts[1]}")
print(f"Ratio patients/témoins: {group_counts[1]/group_counts[0]:.2f}")

# Format pour le document scientifique
print("\nFORMAT POUR DOCUMENT SCIENTIFIQUE:")
print(f"{len(df_valid)} participants ({group_counts[1]} patients avec dépression, {all_n_female} femmes, âge {all_age_mean:.1f} ± {all_age_std:.1f} ans; et {group_counts[0]} témoins sains, {sum(df_valid[df_valid['group'] == 0]['sex'] == 1.0)} femmes, âge {df_valid[df_valid['group'] == 0]['age'].mean():.1f} ± {df_valid[df_valid['group'] == 0]['age'].std():.1f} ans)")
