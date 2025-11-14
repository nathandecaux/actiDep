import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuration
CSV_PATH = '/data/ndecaux/report_hcp_association_50pts_actimetry_calcarine/summary_results.csv'
SELECTED_FEATURES_PATH = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_selected_features.json'
SIGNIFICANT_FEATURES_PATH = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_significant_features.json'
ALL_FEATURES = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_feature_descriptions.json"
SELECTED_METRICS = ['FA','IFW']

OUTPUT_DIR = os.path.dirname(CSV_PATH)

# Filtres de qualité
MAX_REMOVED_SUBJECTS = 0
MAX_REMOVED_POINTS = 2
MIN_SIG_CORRECTED = 1

def load_selected_features(json_path):
    """Charge les features sélectionnées depuis le JSON."""
    with open(json_path, 'r') as f:
        features_dict = json.load(f)
    
    selected_features = [features_dict[key]['name'] for key in features_dict.keys()]
    print(f"Features sélectionnées ({len(selected_features)}): {', '.join(selected_features)}")
    return selected_features, features_dict

def expand_12h_features(selected_features, all_features):
    """Expand features ending with _12h to include all time slices (_12h_1 to _12h_6)."""
    expanded = []
    
    for feature in selected_features:
        if feature.endswith('_12h'):
            # Chercher les variantes avec tranches horaires
            base = feature  # ex: activity_mean_12h
            found_variants = False
            for n in range(1, 7):  # tranches 1 à 6
                variant = f"{base}_{n}"
                if variant in all_features:
                    expanded.append(variant)
                    found_variants = True
            
            if not found_variants:
                # Si aucune variante trouvée, garder l'originale
                expanded.append(feature)
        else:
            expanded.append(feature)
    
    return list(set(expanded))  # Dédupliquer

def load_and_filter_results(csv_path, selected_features, selected_metrics):
    """Charge et filtre les résultats selon les critères de qualité et les sélections."""
    df = pd.read_csv(csv_path)
    
    # Extraction type et var
    if 'type' in df.columns and '_' in df['type'].iloc[0]:
        df['var'] = df['type'].apply(lambda x: '_'.join(x.split('_')[1:]))
        df['type'] = df['type'].apply(lambda x: x.split('_')[0])
    
    # Expansion des features _12h
    all_features = df['var'].unique().tolist()
    expanded_features = expand_12h_features(selected_features, all_features)
    
    print(f"\nFeatures après expansion des tranches 12h:")
    print(f"  Features originales: {len(selected_features)}")
    print(f"  Features après expansion: {len(expanded_features)}")
    
    # Afficher les expansions effectuées
    for orig_feat in selected_features:
        if orig_feat.endswith('_12h'):
            variants = [f for f in expanded_features if f.startswith(orig_feat + '_')]
            if variants:
                print(f"  {orig_feat} -> {', '.join(variants)}")
    
    # Filtrage par features et métriques sélectionnées
    df = df[df['var'].isin(expanded_features)].copy()
    df = df[df['metric'].isin(selected_metrics)].copy()
    
    # Application des filtres de qualité
    mask = (
        (df['removed_subjects'] <= MAX_REMOVED_SUBJECTS) &
        (df['removed_points'] <= MAX_REMOVED_POINTS) &
        (df['n_sig_corrected'] >= MIN_SIG_CORRECTED)
    )
    
    df_filtered = df[mask].copy()
    print(f"\nRésultats chargés : {len(df)} lignes")
    print(f"Après filtrage : {len(df_filtered)} lignes")
    print(f"  - Métriques sélectionnées: {selected_metrics}")
    print(f"  - removed_subjects <= {MAX_REMOVED_SUBJECTS}")
    print(f"  - removed_points <= {MAX_REMOVED_POINTS}")
    print(f"  - n_sig_corrected >= {MIN_SIG_CORRECTED}")
    
    return df_filtered

def summarize_by_bundle_centroid_metric(df):
    """Résume les features par bundle, centroid et métrique."""
    print("\n" + "="*80)
    print("RÉSUMÉ PAR BUNDLE / CENTROID / MÉTRIQUE")
    print("="*80)
    
    # Gestion centroid_id (peut être NaN)
    if 'centroid_id' not in df.columns:
        df['centroid_id'] = np.nan
    
    groupby_cols = ['bundle', 'centroid_id', 'metric']
    grouped = df.groupby(groupby_cols, dropna=False)
    
    summary_rows = []
    
    for (bundle, centroid, metric), group in grouped:
        centroid_str = f"cent{int(centroid)}" if pd.notna(centroid) else "global"
        
        # Compter les features (var) distinctes
        features = group['var'].unique().tolist()
        n_features = len(features)
        
        # Total points significatifs (somme n_sig_corrected)
        total_sig_points = group['n_sig_corrected'].sum()
        
        # Min p-value
        min_p = group['min_p_corrected'].min()
        
        # Max |r| pour corrélations
        max_r = np.nan
        if 'max_abs_r_partial' in group.columns:
            max_r = group['max_abs_r_partial'].max()
        
        summary_rows.append({
            'bundle': bundle,
            'centroid': centroid_str,
            'metric': metric,
            'n_features': n_features,
            'features': ', '.join(sorted(features)),
            'total_sig_points': int(total_sig_points),
            'min_p_corrected': min_p,
            'max_abs_r': max_r
        })
        
        print(f"\n{bundle} / {centroid_str} / {metric}")
        print(f"  Features ({n_features}): {', '.join(sorted(features))}")
        print(f"  Total points significatifs: {int(total_sig_points)}")
        print(f"  Min p-value corrigée: {min_p:.3e}")
        if not np.isnan(max_r):
            print(f"  Max |r| partiel: {max_r:.3f}")
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def load_point_level_data(df, base_dir):
    """Charge les données point par point depuis les CSV détaillés."""
    print("\n" + "="*80)
    print("CHARGEMENT DES DONNÉES POINT PAR POINT")
    print("="*80)
    
    figures_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(figures_dir):
        figures_dir = base_dir
    
    point_data = []
    
    for idx, row in df.iterrows():
        bundle = row['bundle']
        metric = row['metric']
        var = row['var']
        type_analysis = row['type']
        centroid_id = row.get('centroid_id', np.nan)
        
        # Construction du nom de fichier
        if pd.notna(centroid_id):
            centroid_str = f"_cent{int(centroid_id)}"
        else:
            centroid_str = ""
        
        analysis_suffix = 'corrected' if type_analysis == 'group' else 'partial'
        csv_name = f"{bundle}{centroid_str}_{metric}_{var}_{type_analysis}_{analysis_suffix}.csv"
        csv_path = os.path.join(figures_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"  Fichier non trouvé: {csv_name}")
            continue
        
        try:
            point_df = pd.read_csv(csv_path)
            if 'sig_afq' not in point_df.columns or 'point' not in point_df.columns:
                continue
            
            sig_points = point_df[point_df['sig_afq']]['point'].values
            
            for pt in sig_points:
                point_data.append({
                    'bundle': bundle,
                    'centroid_id': centroid_id,
                    'metric': metric,
                    'point': pt,
                    'feature': var,
                    'type': type_analysis
                })
        
        except Exception as e:
            print(f"  Erreur lecture {csv_name}: {e}")
    
    if not point_data:
        print("Aucune donnée point par point chargée.")
        return pd.DataFrame()
    
    point_df = pd.DataFrame(point_data)
    print(f"Total lignes chargées: {len(point_df)}")
    return point_df

def summarize_by_bundle_centroid_point(point_df):
    """Compte le nombre de features significatives par bundle/centroid/point."""
    print("\n" + "="*80)
    print("NOMBRE DE FEATURES PAR BUNDLE / CENTROID / POINT")
    print("="*80)
    
    if point_df.empty:
        print("Aucune donnée à résumer.")
        return pd.DataFrame()
    
    groupby_cols = ['bundle', 'centroid_id', 'point']
    grouped = point_df.groupby(groupby_cols, dropna=False)
    
    summary_rows = []
    
    for (bundle, centroid, point), group in grouped:
        centroid_str = f"cent{int(centroid)}" if pd.notna(centroid) else "global"
        n_features = group['feature'].nunique()
        features = sorted(group['feature'].unique())
        
        summary_rows.append({
            'bundle': bundle,
            'centroid': centroid_str,
            'point': int(point),
            'n_features': n_features,
            'features': ', '.join(features)
        })
    
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['bundle', 'centroid', 'point']
    )
    
    # Affichage des top points (plus de features)
    top_points = summary_df.nlargest(20, 'n_features')
    print("\nTop 20 points avec le plus de features significatives:")
    for _, row in top_points.iterrows():
        print(f"  {row['bundle']} / {row['centroid']} / point {row['point']}: "
              f"{row['n_features']} features ({row['features']})")
    
    return summary_df

def summarize_by_point_metric(point_df):
    """Compte le nombre de features significatives par point/métrique."""
    print("\n" + "="*80)
    print("NOMBRE DE FEATURES PAR POINT / MÉTRIQUE (TOUS BUNDLES)")
    print("="*80)
    
    if point_df.empty:
        print("Aucune donnée à résumer.")
        return pd.DataFrame()
    
    grouped = point_df.groupby(['point', 'metric'], dropna=False)
    
    summary_rows = []
    
    for (point, metric), group in grouped:
        n_features = group['feature'].nunique()
        n_bundles = group['bundle'].nunique()
        
        summary_rows.append({
            'point': int(point),
            'metric': metric,
            'n_features': n_features,
            'n_bundles': n_bundles
        })
    
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['metric', 'point']
    )
    
    # Affichage par métrique
    for metric in summary_df['metric'].unique():
        metric_df = summary_df[summary_df['metric'] == metric]
        top = metric_df.nlargest(10, 'n_features')
        print(f"\nMétrique {metric} - Top 10 points:")
        for _, row in top.iterrows():
            print(f"  Point {row['point']}: {row['n_features']} features "
                  f"({row['n_bundles']} bundles)")
    
    return summary_df

def plot_point_frequency(point_df, output_dir):
    """Crée un graphique de fréquence de significativité par point pour chaque bundle/centroid/métrique."""
    print("\n" + "="*80)
    print("GÉNÉRATION DES GRAPHIQUES DE FRÉQUENCE PAR POINT")
    print("="*80)
    
    if point_df.empty:
        print("Aucune donnée pour les graphiques.")
        return
    
    plots_dir = os.path.join(output_dir, 'frequency_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Grouper par bundle, centroid_id, metric
    groupby_cols = ['bundle', 'centroid_id', 'metric']
    grouped = point_df.groupby(groupby_cols, dropna=False)
    
    plot_count = 0
    
    for (bundle, centroid, metric), group in grouped:
        centroid_str = f"cent{int(centroid)}" if pd.notna(centroid) else "global"
        
        # Compter le nombre de features significatives par point
        point_counts = group.groupby('point')['feature'].nunique().reset_index()
        point_counts.columns = ['point', 'n_features']
        
        if point_counts.empty:
            continue
        
        # Créer un DataFrame complet avec tous les points (0 à max)
        all_points = pd.DataFrame({'point': range(int(point_counts['point'].min()), 
                                                   int(point_counts['point'].max()) + 1)})
        point_counts = all_points.merge(point_counts, on='point', how='left').fillna(0)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(point_counts['point'], point_counts['n_features'], 
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Point', fontsize=12)
        ax.set_ylabel('Nombre de features significatives', fontsize=12)
        ax.set_title(f'{bundle} / {centroid_str} / {metric}\nFréquence de significativité par point', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Annoter les points avec le plus de features (seulement si > 0)
        top_n = 5
        top_points = point_counts[point_counts['n_features'] > 0].nlargest(top_n, 'n_features')
        for _, row in top_points.iterrows():
            ax.text(row['point'], row['n_features'], 
                   f"{int(row['n_features'])}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"{bundle}_{centroid_str}_{metric}_frequency.png"
        output_path = os.path.join(plots_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        plot_count += 1
        if plot_count % 10 == 0:
            print(f"  {plot_count} graphiques générés...")
    
    print(f"\nTotal graphiques générés: {plot_count}")
    print(f"Sauvegardés dans: {plots_dir}")

def plot_cumulative_frequency(point_df, output_dir):
    """Crée des graphiques cumulés (toutes métriques) par bundle/centroid."""
    print("\n" + "="*80)
    print("GÉNÉRATION DES GRAPHIQUES CUMULÉS PAR BUNDLE/CENTROID")
    print("="*80)
    
    if point_df.empty:
        print("Aucune donnée pour les graphiques cumulés.")
        return
    
    plots_dir = os.path.join(output_dir, 'frequency_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Grouper par bundle et centroid_id (sans métrique)
    groupby_cols = ['bundle', 'centroid_id']
    grouped = point_df.groupby(groupby_cols, dropna=False)
    
    plot_count = 0
    
    for (bundle, centroid), group in grouped:
        centroid_str = f"cent{int(centroid)}" if pd.notna(centroid) else "global"
        
        # Compter par point ET métrique
        point_metric_counts = group.groupby(['point', 'metric'])['feature'].nunique().reset_index()
        point_metric_counts.columns = ['point', 'metric', 'n_features']
        
        # Plage complète de points
        min_pt = int(point_metric_counts['point'].min())
        max_pt = int(point_metric_counts['point'].max())
        all_points = range(min_pt, max_pt + 1)
        
        # Créer figure avec barres empilées
        fig, ax = plt.subplots(figsize=(14, 6))
        
        metrics = sorted(point_metric_counts['metric'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        
        # Préparer les données pour barres empilées
        bottom = np.zeros(len(all_points))
        point_to_idx = {pt: i for i, pt in enumerate(all_points)}
        
        for metric, color in zip(metrics, colors):
            metric_data = point_metric_counts[point_metric_counts['metric'] == metric]
            heights = np.zeros(len(all_points))
            
            for _, row in metric_data.iterrows():
                idx = point_to_idx[int(row['point'])]
                heights[idx] = row['n_features']
            
            ax.bar(all_points, heights, bottom=bottom, 
                   label=metric, color=color, alpha=0.8, edgecolor='black', linewidth=0.3)
            bottom += heights
        
        ax.set_xlabel('Point', fontsize=12)
        ax.set_ylabel('Nombre total de features significatives', fontsize=12)
        ax.set_title(f'{bundle} / {centroid_str}\nFréquence cumulée (toutes métriques)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(title='Métrique', loc='upper right', fontsize=9)
        
        # Annoter les pics (top 5 points avec le plus de features cumulées)
        cumulative_counts = pd.DataFrame({
            'point': all_points,
            'total': bottom
        })
        top_points = cumulative_counts[cumulative_counts['total'] > 0].nlargest(5, 'total')
        for _, row in top_points.iterrows():
            ax.text(row['point'], row['total'], 
                   f"{int(row['total'])}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"{bundle}_{centroid_str}_cumulative_frequency.png"
        output_path = os.path.join(plots_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        plot_count += 1
        if plot_count % 10 == 0:
            print(f"  {plot_count} graphiques cumulés générés...")
    
    print(f"\nTotal graphiques cumulés générés: {plot_count}")
    print(f"Sauvegardés dans: {plots_dir}")

def load_correlation_values(df, base_dir):
    """Charge les valeurs de corrélation depuis les CSV détaillés."""
    print("\n" + "="*80)
    print("CHARGEMENT DES VALEURS DE CORRÉLATION")
    print("="*80)
    
    figures_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(figures_dir):
        figures_dir = base_dir
    
    correlation_data = []
    
    for idx, row in df.iterrows():
        bundle = row['bundle']
        metric = row['metric']
        var = row['var']
        type_analysis = row['type']
        centroid_id = row.get('centroid_id', np.nan)
        
        # Construction du nom de fichier
        if pd.notna(centroid_id):
            centroid_str = f"_cent{int(centroid_id)}"
        else:
            centroid_str = ""
        
        analysis_suffix = 'corrected' if type_analysis == 'group' else 'partial'
        csv_name = f"{bundle}{centroid_str}_{metric}_{var}_{type_analysis}_{analysis_suffix}.csv"
        csv_path = os.path.join(figures_dir, csv_name)
        
        if not os.path.exists(csv_path):
            print(f"  Fichier non trouvé: {csv_name}")
            continue
        
        try:
            point_df = pd.read_csv(csv_path)
            
            # Chercher la colonne de corrélation (r, r_partial, etc.)
            corr_col = None
            for col in ['r_partial', 'r', 'correlation']:
                if col in point_df.columns:
                    corr_col = col
                    break
            
            if corr_col is None or 'sig_afq' not in point_df.columns:
                continue
            
            # Filtrer uniquement les points significatifs
            sig_data = point_df[point_df['sig_afq']].copy()
            
            if len(sig_data) == 0:
                continue
            
            # Calculer min, mean, max des corrélations
            correlation_data.append({
                'bundle': bundle,
                'centroid_id': centroid_id,
                'metric': metric,
                'feature': var,
                'type': type_analysis,
                'min_corr': sig_data[corr_col].min(),
                'mean_corr': sig_data[corr_col].mean(),
                'max_corr': sig_data[corr_col].max(),
                'n_sig_points': len(sig_data)
            })
        
        except Exception as e:
            print(f"  Erreur lecture {csv_name}: {e}")
    
    if not correlation_data:
        print("Aucune donnée de corrélation chargée.")
        return pd.DataFrame()
    
    corr_df = pd.DataFrame(correlation_data)
    print(f"Total lignes chargées: {len(corr_df)}")
    return corr_df

def plot_correlation_matrices(corr_df, output_dir, features_dict):
    """Crée des matrices de corrélation par métrique."""
    print("\n" + "="*80)
    print("GÉNÉRATION DES MATRICES DE CORRÉLATION")
    print("="*80)
    
    if corr_df.empty:
        print("Aucune donnée pour les matrices.")
        return
    
    plots_dir = os.path.join(output_dir, 'correlation_matrices')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculer le nombre de centroids par bundle
    bundle_centroid_counts = {}
    for bundle in corr_df['bundle'].unique():
        bundle_data = corr_df[corr_df['bundle'] == bundle]
        valid_centroids = bundle_data[bundle_data['centroid_id'].notna()]['centroid_id'].unique()
        if len(valid_centroids) > 0:
            bundle_centroid_counts[bundle] = int(valid_centroids.max())+1
    
    # Créer une colonne bundle_centroid pour l'axe Y
    def format_bundle_centroid(row):
        bundle = row['bundle']
        if pd.notna(row['centroid_id']):
            centroid_num = int(row['centroid_id'])
            if bundle in bundle_centroid_counts:
                total_centroids = bundle_centroid_counts[bundle]
                return f"{bundle} ({centroid_num+1}/{total_centroids})"
            return f"{bundle} ({centroid_num})"
        return f"{bundle} (global)"
    
    corr_df['bundle_centroid'] = corr_df.apply(format_bundle_centroid, axis=1)
    
    # Créer un mapping feature -> expected_correlation par métrique
    feature_expected_corr = {}
    for key, feat_info in features_dict.items():
        feat_name = feat_info['name']
        if 'excepted_correlation' in feat_info:
            feature_expected_corr[feat_name] = feat_info['excepted_correlation']
    
    # Pour chaque métrique
    for metric in sorted(corr_df['metric'].unique()):
        metric_data = corr_df[corr_df['metric'] == metric].copy()
        
        # Créer le pivot table (moyenne des corrélations moyennes)
        # Si plusieurs types d'analyse, on prend la moyenne
        pivot_data = metric_data.groupby(['bundle_centroid', 'feature'])['mean_corr'].mean().reset_index()
        pivot_matrix = pivot_data.pivot(index='bundle_centroid', columns='feature', values='mean_corr')
        
        # Trier les colonnes (features) et lignes (bundle_centroid)
        pivot_matrix = pivot_matrix.sort_index(axis=0).sort_index(axis=1)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(max(14, len(pivot_matrix.columns) * 0.8), 
                                        max(8, len(pivot_matrix) * 0.4)))
        
        # Utiliser une palette divergente centrée sur 0
        vmax = max(abs(pivot_matrix.min().min()), abs(pivot_matrix.max().max()))
        vmin = -vmax
        
        # Créer une matrice d'annotations avec valeur + signe attendu
        annot_matrix = pivot_matrix.copy()
        for col in annot_matrix.columns:
            # Extraire le nom de base de la feature (sans _1, _2, etc.)
            base_feat = col
            for n in range(1, 7):
                if col.endswith(f'_12h_{n}'):
                    base_feat = col.replace(f'_12h_{n}', '_12h')
                    break
            
            expected_sign = None
            if base_feat in feature_expected_corr:
                expected_corr = feature_expected_corr[base_feat]
                if metric in expected_corr:
                    expected_sign = expected_corr[metric]
            
            # Formater les annotations
            for idx in annot_matrix.index:
                val = annot_matrix.loc[idx, col]
                if pd.notna(val):
                    if expected_sign:
                        annot_matrix.loc[idx, col] = f"{val:.3f}\n({expected_sign})"
                    else:
                        annot_matrix.loc[idx, col] = f"{val:.3f}"
        
        sns.heatmap(pivot_matrix, 
                    annot=annot_matrix, 
                    fmt='', 
                    cmap='RdBu_r', 
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'label': 'Corrélation moyenne'},
                    linewidths=0.5,
                    linecolor='gray',
                    ax=ax,
                    annot_kws={'size': 7})
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bundle (N° centroid)', fontsize=12, fontweight='bold')
        ax.set_title(f'Matrice de corrélation - {metric}\nMoyenne des corrélations sur points significatifs\n(Signe attendu entre parenthèses)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"correlation_matrix_{metric}.png"
        output_path = os.path.join(plots_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Matrice générée pour {metric}: {filename}")
        
        # Sauvegarder aussi en CSV avec signe attendu
        csv_data = []
        for idx in pivot_matrix.index:
            row_data = {'bundle_centroid': idx}
            for col in pivot_matrix.columns:
                val = pivot_matrix.loc[idx, col]
                
                # Extraire le nom de base de la feature
                base_feat = col
                for n in range(1, 7):
                    if col.endswith(f'_12h_{n}'):
                        base_feat = col.replace(f'_12h_{n}', '_12h')
                        break
                
                expected_sign = None
                if base_feat in feature_expected_corr:
                    expected_corr = feature_expected_corr[base_feat]
                    if metric in expected_corr:
                        expected_sign = expected_corr[metric]
                
                if pd.notna(val):
                    if expected_sign:
                        row_data[col] = f"{val:.4f} ({expected_sign})"
                    else:
                        row_data[col] = f"{val:.4f}"
                else:
                    row_data[col] = ""
            csv_data.append(row_data)
        
        csv_df = pd.DataFrame(csv_data)
        csv_filename = f"correlation_matrix_{metric}.csv"
        csv_path = os.path.join(plots_dir, csv_filename)
        csv_df.to_csv(csv_path, index=False)
        print(f"  CSV sauvegardé: {csv_filename}")
    
    print(f"\nMatrices sauvegardées dans: {plots_dir}")

def group_features_by_prefix(features):
    """Groupe les features par préfixe (premier mot avant _)."""
    groups = {}
    for feature in features:
        prefix = feature.split('_')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(feature)
    return groups

def plot_correlation_matrices_by_group(corr_df, output_dir, features_dict, feature_set_name='selected'):
    """Crée des matrices de corrélation par métrique et par groupe de features."""
    print("\n" + "="*80)
    print(f"GÉNÉRATION DES MATRICES DE CORRÉLATION - {feature_set_name.upper()}")
    print("="*80)
    
    if corr_df.empty:
        print("Aucune donnée pour les matrices.")
        return
    
    plots_dir = os.path.join(output_dir, 'correlation_matrices', feature_set_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculer le nombre de centroids par bundle
    bundle_centroid_counts = {}
    for bundle in corr_df['bundle'].unique():
        bundle_data = corr_df[corr_df['bundle'] == bundle]
        valid_centroids = bundle_data[bundle_data['centroid_id'].notna()]['centroid_id'].unique()
        if len(valid_centroids) > 0:
            bundle_centroid_counts[bundle] = int(valid_centroids.max())+1
    
    # Créer une colonne bundle_centroid pour l'axe Y
    def format_bundle_centroid(row):
        bundle = row['bundle']
        if pd.notna(row['centroid_id']):
            centroid_num = int(row['centroid_id'])
            if bundle in bundle_centroid_counts:
                total_centroids = bundle_centroid_counts[bundle]
                return f"{bundle} ({centroid_num+1}/{total_centroids})"
            return f"{bundle} ({centroid_num})"
        return f"{bundle} (global)"
    
    corr_df['bundle_centroid'] = corr_df.apply(format_bundle_centroid, axis=1)
    
    # Créer un mapping feature -> expected_correlation par métrique
    feature_expected_corr = {}
    for key, feat_info in features_dict.items():
        feat_name = feat_info['name']
        if 'excepted_correlation' in feat_info:
            feature_expected_corr[feat_name] = feat_info['excepted_correlation']
    
    # Pour chaque métrique
    for metric in sorted(corr_df['metric'].unique()):
        metric_data = corr_df[corr_df['metric'] == metric].copy()
        
        # Grouper les features par préfixe
        all_features = metric_data['feature'].unique().tolist()
        feature_groups = group_features_by_prefix(all_features)
        
        print(f"\nMétrique {metric}: {len(feature_groups)} groupes de features")
        
        # Créer une matrice par groupe
        for group_name, group_features in feature_groups.items():
            print(f"  Groupe {group_name}: {len(group_features)} features")
            
            # Filtrer les données pour ce groupe
            group_data = metric_data[metric_data['feature'].isin(group_features)].copy()
            
            if group_data.empty:
                continue
            
            # Créer le pivot table
            pivot_data = group_data.groupby(['bundle_centroid', 'feature'])['mean_corr'].mean().reset_index()
            pivot_matrix = pivot_data.pivot(index='bundle_centroid', columns='feature', values='mean_corr')
            
            # Trier les colonnes et lignes
            pivot_matrix = pivot_matrix.sort_index(axis=0).sort_index(axis=1)
            
            # Créer le graphique
            fig, ax = plt.subplots(figsize=(max(12, len(pivot_matrix.columns) * 0.6), 
                                            max(8, len(pivot_matrix) * 0.4)))
            
            # Utiliser une palette divergente centrée sur 0
            vmax = max(abs(pivot_matrix.min().min()), abs(pivot_matrix.max().max()))
            vmin = -vmax
            
            # Créer une matrice d'annotations avec valeur + signe attendu
            annot_matrix = pivot_matrix.copy()
            for col in annot_matrix.columns:
                # Extraire le nom de base de la feature
                base_feat = col
                for n in range(1, 7):
                    if col.endswith(f'_12h_{n}'):
                        base_feat = col.replace(f'_12h_{n}', '_12h')
                        break
                
                expected_sign = None
                if base_feat in feature_expected_corr:
                    expected_corr = feature_expected_corr[base_feat]
                    if metric in expected_corr:
                        expected_sign = expected_corr[metric]
                
                # Formater les annotations
                for idx in annot_matrix.index:
                    val = annot_matrix.loc[idx, col]
                    if pd.notna(val):
                        if expected_sign:
                            annot_matrix.loc[idx, col] = f"{val:.3f}\n({expected_sign})"
                        else:
                            annot_matrix.loc[idx, col] = f"{val:.3f}"
            
            sns.heatmap(pivot_matrix, 
                        annot=annot_matrix, 
                        fmt='', 
                        cmap='RdBu_r', 
                        center=0,
                        vmin=vmin,
                        vmax=vmax,
                        cbar_kws={'label': 'Corrélation moyenne'},
                        linewidths=0.5,
                        linecolor='gray',
                        ax=ax,
                        annot_kws={'size': 7})
            
            ax.set_xlabel('Features', fontsize=12, fontweight='bold')
            ax.set_ylabel('Bundle (N° centroid)', fontsize=12, fontweight='bold')
            ax.set_title(f'Matrice de corrélation - {metric} - Groupe: {group_name}\n'
                        f'Moyenne des corrélations sur points significatifs\n'
                        f'(Signe attendu entre parenthèses)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            
            # Sauvegarder
            filename = f"correlation_matrix_{metric}_{group_name}.png"
            output_path = os.path.join(plots_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Sauvegarder aussi en CSV avec signe attendu
            csv_data = []
            for idx in pivot_matrix.index:
                row_data = {'bundle_centroid': idx}
                for col in pivot_matrix.columns:
                    val = pivot_matrix.loc[idx, col]
                    
                    # Extraire le nom de base de la feature
                    base_feat = col
                    for n in range(1, 7):
                        if col.endswith(f'_12h_{n}'):
                            base_feat = col.replace(f'_12h_{n}', '_12h')
                            break
                    
                    expected_sign = None
                    if base_feat in feature_expected_corr:
                        expected_corr = feature_expected_corr[base_feat]
                        if metric in expected_corr:
                            expected_sign = expected_corr[metric]
                    
                    if pd.notna(val):
                        if expected_sign:
                            row_data[col] = f"{val:.4f} ({expected_sign})"
                        else:
                            row_data[col] = f"{val:.4f}"
                    else:
                        row_data[col] = ""
                csv_data.append(row_data)
            
            csv_df = pd.DataFrame(csv_data)
            csv_filename = f"correlation_matrix_{metric}_{group_name}.csv"
            csv_path = os.path.join(plots_dir, csv_filename)
            csv_df.to_csv(csv_path, index=False)
    
    print(f"\nMatrices sauvegardées dans: {plots_dir}")

def summarize_correlations(corr_df, output_dir, features_dict):
    """Résume les statistiques de corrélation."""
    print("\n" + "="*80)
    print("RÉSUMÉ DES CORRÉLATIONS")
    print("="*80)
    
    if corr_df.empty:
        print("Aucune donnée à résumer.")
        return pd.DataFrame()
    
    # Créer un mapping feature -> expected_correlation par métrique
    feature_expected_corr = {}
    for key, feat_info in features_dict.items():
        feat_name = feat_info['name']
        if 'excepted_correlation' in feat_info:
            feature_expected_corr[feat_name] = feat_info['excepted_correlation']
    
    summary_rows = []
    
    groupby_cols = ['bundle', 'centroid_id', 'metric', 'feature']
    grouped = corr_df.groupby(groupby_cols, dropna=False)
    
    for (bundle, centroid, metric, feature), group in grouped:
        centroid_str = f"cent{int(centroid)}" if pd.notna(centroid) else "global"
        
        # Extraire le nom de base de la feature
        base_feat = feature
        for n in range(1, 7):
            if feature.endswith(f'_12h_{n}'):
                base_feat = feature.replace(f'_12h_{n}', '_12h')
                break
        
        expected_sign = None
        if base_feat in feature_expected_corr:
            expected_corr = feature_expected_corr[base_feat]
            if metric in expected_corr:
                expected_sign = expected_corr[metric]
        
        mean_corr = group['mean_corr'].mean()
        
        # Vérifier si la corrélation observée correspond au signe attendu
        match_expected = None
        if expected_sign:
            if expected_sign == '+' and mean_corr > 0:
                match_expected = 'OUI'
            elif expected_sign == '-' and mean_corr < 0:
                match_expected = 'OUI'
            else:
                match_expected = 'NON'
        
        summary_rows.append({
            'bundle': bundle,
            'centroid': centroid_str,
            'metric': metric,
            'feature': feature,
            'expected_sign': expected_sign if expected_sign else 'N/A',
            'min_corr': group['min_corr'].min(),
            'mean_corr': mean_corr,
            'max_corr': group['max_corr'].max(),
            'match_expected': match_expected if match_expected else 'N/A',
            'total_sig_points': group['n_sig_points'].sum()
        })
    
    summary_df = pd.DataFrame(summary_rows).sort_values(['metric', 'bundle', 'centroid', 'feature'])
    
    # Sauvegarder
    out_path = os.path.join(output_dir, 'correlation_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"Résumé des corrélations sauvegardé: {out_path}")
    
    # Afficher quelques statistiques
    print("\nStatistiques globales par métrique:")
    for metric in summary_df['metric'].unique():
        metric_data = summary_df[summary_df['metric'] == metric]
        print(f"\n{metric}:")
        print(f"  Corrélation moyenne (absolue): {metric_data['mean_corr'].abs().mean():.3f}")
        print(f"  Corrélation max (absolue): {metric_data['max_corr'].abs().max():.3f}")
        print(f"  Nombre de combinaisons: {len(metric_data)}")
        
        # Statistiques de concordance avec signe attendu
        matches = metric_data[metric_data['match_expected'] != 'N/A']
        if len(matches) > 0:
            n_match = len(matches[matches['match_expected'] == 'OUI'])
            pct_match = (n_match / len(matches)) * 100
            print(f"  Concordance avec signe attendu: {n_match}/{len(matches)} ({pct_match:.1f}%)")
    
    return summary_df

def generate_summary_report(csv_path, output_dir=None):
    """Génère un rapport complet de résumé."""
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Créer une structure de dossiers organisée
    results_dir = os.path.join(output_dir, 'tractometry_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    summaries_dir = os.path.join(results_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    
    # 0. Charger les features sélectionnées
    selected_features, features_dict_selected = load_selected_features(SELECTED_FEATURES_PATH)
    
    # Charger toutes les features
    with open(ALL_FEATURES, 'r') as f:
        all_features_dict = json.load(f)
    all_feature_names = [all_features_dict[key]['name'] for key in all_features_dict.keys()]
    
    # 1. Analyse avec features sélectionnées
    print("\n" + "="*80)
    print("ANALYSE AVEC FEATURES SÉLECTIONNÉES")
    print("="*80)
    
    df_filtered_selected = load_and_filter_results(csv_path, selected_features, SELECTED_METRICS)
    
    if not df_filtered_selected.empty:
        # Chargement des valeurs de corrélation
        corr_df_selected = load_correlation_values(df_filtered_selected, os.path.dirname(csv_path))
        
        if not corr_df_selected.empty:
            # Résumé des corrélations
            correlation_summary_selected = summarize_correlations(corr_df_selected, summaries_dir, features_dict_selected)
            
            # Génération des matrices de corrélation par groupe
            plot_correlation_matrices_by_group(corr_df_selected, results_dir, features_dict_selected, 'selected')
            
            # Résumé par bundle/centroid/métrique
            summary_bcm_selected = summarize_by_bundle_centroid_metric(df_filtered_selected)
            out_bcm = os.path.join(summaries_dir, 'selected_summary_by_bundle_centroid_metric.csv')
            summary_bcm_selected.to_csv(out_bcm, index=False)
            print(f"\nRésumé bundle/centroid/métrique (selected) sauvegardé: {out_bcm}")
            
            # Chargement données point par point
            point_df_selected = load_point_level_data(df_filtered_selected, os.path.dirname(csv_path))
            
            if not point_df_selected.empty:
                # Résumé par bundle/centroid/point
                summary_bcp_selected = summarize_by_bundle_centroid_point(point_df_selected)
                out_bcp = os.path.join(summaries_dir, 'selected_summary_by_bundle_centroid_point.csv')
                summary_bcp_selected.to_csv(out_bcp, index=False)
                
                # Résumé par point/métrique
                summary_pm_selected = summarize_by_point_metric(point_df_selected)
                out_pm = os.path.join(summaries_dir, 'selected_summary_by_point_metric.csv')
                summary_pm_selected.to_csv(out_pm, index=False)
                
                # Génération des graphiques de fréquence
                freq_plots_dir = os.path.join(results_dir, 'frequency_plots', 'selected')
                os.makedirs(freq_plots_dir, exist_ok=True)
                plot_point_frequency(point_df_selected, freq_plots_dir)
                plot_cumulative_frequency(point_df_selected, freq_plots_dir)
    
    # 2. Analyse avec TOUTES les features
    print("\n" + "="*80)
    print("ANALYSE AVEC TOUTES LES FEATURES")
    print("="*80)
    
    df_filtered_all = load_and_filter_results(csv_path, all_feature_names, SELECTED_METRICS)
    
    if not df_filtered_all.empty:
        # Chargement des valeurs de corrélation
        corr_df_all = load_correlation_values(df_filtered_all, os.path.dirname(csv_path))
        
        if not corr_df_all.empty:
            # Résumé des corrélations
            correlation_summary_all = summarize_correlations(corr_df_all, summaries_dir, all_features_dict)
            out_corr_all = os.path.join(summaries_dir, 'all_features_correlation_summary.csv')
            correlation_summary_all.to_csv(out_corr_all, index=False)
            
            # Génération des matrices de corrélation par groupe
            plot_correlation_matrices_by_group(corr_df_all, results_dir, all_features_dict, 'all_features')
            
            # Résumé par bundle/centroid/métrique
            summary_bcm_all = summarize_by_bundle_centroid_metric(df_filtered_all)
            out_bcm_all = os.path.join(summaries_dir, 'all_features_summary_by_bundle_centroid_metric.csv')
            summary_bcm_all.to_csv(out_bcm_all, index=False)
            print(f"\nRésumé bundle/centroid/métrique (all) sauvegardé: {out_bcm_all}")
            
            # Chargement données point par point
            point_df_all = load_point_level_data(df_filtered_all, os.path.dirname(csv_path))
            
            if not point_df_all.empty:
                # Résumé par bundle/centroid/point
                summary_bcp_all = summarize_by_bundle_centroid_point(point_df_all)
                out_bcp_all = os.path.join(summaries_dir, 'all_features_summary_by_bundle_centroid_point.csv')
                summary_bcp_all.to_csv(out_bcp_all, index=False)
                
                # Résumé par point/métrique
                summary_pm_all = summarize_by_point_metric(point_df_all)
                out_pm_all = os.path.join(summaries_dir, 'all_features_summary_by_point_metric.csv')
                summary_pm_all.to_csv(out_pm_all, index=False)
                
                # Génération des graphiques de fréquence
                freq_plots_dir_all = os.path.join(results_dir, 'frequency_plots', 'all_features')
                os.makedirs(freq_plots_dir_all, exist_ok=True)
                plot_point_frequency(point_df_all, freq_plots_dir_all)
                plot_cumulative_frequency(point_df_all, freq_plots_dir_all)
    
    # 3. Génération du rapport Markdown global
    md_lines = [
        "# Résumé des résultats de tractométrie",
        "",
        f"Généré le: {pd.Timestamp.now()}",
        "",
        "## Structure du dossier de résultats",
        "",
        "```",
        "tractometry_analysis/",
        "├── correlation_matrices/",
        "│   ├── selected/          # Matrices pour features sélectionnées",
        "│   │   ├── correlation_matrix_FA_<group>.png",
        "│   │   ├── correlation_matrix_FA_<group>.csv",
        "│   │   └── ...",
        "│   └── all_features/      # Matrices pour toutes les features",
        "│       ├── correlation_matrix_FA_<group>.png",
        "│       ├── correlation_matrix_FA_<group>.csv",
        "│       └── ...",
        "├── frequency_plots/",
        "│   ├── selected/          # Graphiques de fréquence (features sélectionnées)",
        "│   └── all_features/      # Graphiques de fréquence (toutes features)",
        "├── summaries/             # Fichiers CSV de résumé",
        "│   ├── selected_*.csv",
        "│   └── all_features_*.csv",
        "└── tractometry_summary.md # Ce fichier",
        "```",
        "",
        "## Critères de filtrage",
        f"- Métriques sélectionnées: {', '.join(SELECTED_METRICS)}",
        f"- Sujets retirés: <= {MAX_REMOVED_SUBJECTS}",
        f"- Points retirés: <= {MAX_REMOVED_POINTS}",
        f"- Points significatifs (corrigés): >= {MIN_SIG_CORRECTED}",
        "",
        "## Analyse avec features sélectionnées",
        f"- Nombre de features: {len(selected_features)}",
        f"- Résultats filtrés: {len(df_filtered_selected) if not df_filtered_selected.empty else 0}",
        "",
        "## Analyse avec toutes les features",
        f"- Nombre de features: {len(all_feature_names)}",
        f"- Résultats filtrés: {len(df_filtered_all) if not df_filtered_all.empty else 0}",
        "",
        "## Groupes de features",
        "",
    ]
    
    # Lister les groupes de features
    if not corr_df_all.empty:
        all_features_list = corr_df_all['feature'].unique().tolist()
        feature_groups = group_features_by_prefix(all_features_list)
        
        md_lines.append("| Groupe | Nombre de features |")
        md_lines.append("|--------|-------------------|")
        for group_name, group_features in sorted(feature_groups.items()):
            md_lines.append(f"| {group_name} | {len(group_features)} |")
        md_lines.append("")
    
    md_lines.extend([
        "## Matrices de corrélation",
        "",
        "Les matrices sont organisées par:",
        "- **Métrique** (FA, IFW)",
        "- **Groupe de features** (préfixe: activity, freq, walk, oadl, inactivity)",
        "- **Type d'analyse** (selected vs all_features)",
        "",
        "Chaque matrice affiche:",
        "- Les valeurs de corrélation moyenne sur les points significatifs",
        "- Le signe de corrélation attendu entre parenthèses (+/-)",
        "- Les bundles en lignes avec numéro de centroid",
        "- Les features en colonnes",
        ""
    ])
    
    # Sauvegarder le rapport
    out_md = os.path.join(results_dir, 'tractometry_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\nRapport Markdown sauvegardé: {out_md}")
    print("\n" + "="*80)
    print("RÉSUMÉ TERMINÉ")
    print(f"Tous les résultats sont dans: {results_dir}")
    print("="*80)

if __name__ == '__main__':
    generate_summary_report(CSV_PATH, OUTPUT_DIR)
