import os
from os.path import join as opj
import pandas as pd
import glob
import re
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import panel as pn
# pn.extension('bokeh') # Already called by hv.extension('bokeh')
import param
import random  # Ajouter cette importation en haut du fichier
import numpy as np
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression 
from tractseg.libs.AFQ_MultiCompCorrection import get_significant_areas  # supprimé pour corrélations -> ne plus utilisé
from tractseg.libs import metric_utils
from collections import defaultdict
from actiDep.analysis.afq_optimized import AFQ_MultiCompCorrection
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.interpolation import map_coordinates
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa

from tractseg.libs import fiber_utils
from tqdm import tqdm

#### Custom imports ####
from pprint import pprint
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import sys
import pathlib
from subprocess import call
from actiDep.set_config import set_config,get_HCP_bundle_names
from actiDep.data.loader import Subject, parse_filename, ActiDepFile, Actidep
from actiDep.utils.tools import del_key, upt_dict, add_kwargs_to_cli, run_cli_command, run_mrtrix_command
import SimpleITK as sitk
import json
import tempfile
import glob
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as sp_stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# db_root, ds, csv_files, bundle_names, corr_variables, classif_variables, confond lists

db_root = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'

cache = False
#If ds_association.pkl already exists, load it instead of re-running the whole pipeline

ds = Actidep(db_root)

csv_files = ds.get_global(pipeline='hcp_association_24pts',extension='csv',datatype='metric')

# from pprint import pprint

# pprint([f.path for f in csv_files if f.get_entities()['bundle']=='CSTleft'])

bundle_names = list(get_HCP_bundle_names().keys())
#Filter for now to only look at CSTleft
# bundle_names = ['CSTleft','CSTright']

corr_variables = ['ami','aes']

actimetry = pd.read_excel('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_features.xlsx')
#Get all column names from actimetry except subject_id and participant_id
actimetry_columns = [col for col in actimetry.columns if col not in ['subject_id','participant_id']]

# corr_variables += actimetry_columns

classif_variables = {'group':'with_controls','apathy':'no_controls'}

confond_variables_with_control = ['age','sex','city']

confond_variables_without_control = confond_variables_with_control + ['duration_dep','type_dep']


def load_and_merge_bundle_csvs(bundle_name, bundle_csvs):
    metric_files_dict = {f.get_full_entities()['subject']: f for f in bundle_csvs}

    #Load all csv files
    metric_files = [pd.read_csv(f.path) for f in bundle_csvs]
    #For all dataframe, add a column with the subject id
    for df, f in zip(metric_files, bundle_csvs):
        df['subject'] = f.get_full_entities()['subject']
        df['participant_id'] = 'sub-' + df["subject"].astype(str)  # Assuming subject is a string of digits
    #Concatenate all dataframes
    metrics_df = pd.concat(metric_files, ignore_index=True)

    additional_info = pd.read_excel("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx")
    #Merge the additional info with the metrics dataframe using subject <-> participant_id as key
    metrics_df = metrics_df.merge(additional_info, on='participant_id', how='left')

    actimetry = pd.read_excel('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_features.xlsx')
    metrics_df = metrics_df.merge(actimetry, on='participant_id', how='left')
    
    return metrics_df, metric_files_dict


for bundle_name in bundle_names:
    bundle_csvs = [f for f in csv_files if f.get_entities()['bundle']==bundle_name]
    
    bundle_df , bundle_metric_files_dict= load_and_merge_bundle_csvs(bundle_name, bundle_csvs)

import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, pearsonr
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

METRIC_COLUMNS_CANDIDATES = ['FA','MD','RD','AD','IFW','IRF']
CLASSIF_CONFOUND_MAP = {
    'group': confond_variables_with_control,
    'apathy': confond_variables_without_control
}

AFQ_ALPHA = 0.05
AFQ_NPERM = 5000

# Nouveau: activer correction multi-bundle (AFQ une seule fois par métrique)
MULTI_BUNDLE_AFQ = False  # Equivalent du --mc de plot_tractometry_results

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def detect_point_column(df):
    if 'point_id' in df.columns:
        return 'point_id'
    if 'point' in df.columns:
        return 'point'
    # fallback: try to infer
    for c in df.columns:
        if re.fullmatch(r'point(_?id)?', c.lower()):
            return c
    raise ValueError("Colonne des points introuvable (point_id / point).")

def detect_metric_columns(df):
    return [c for c in METRIC_COLUMNS_CANDIDATES if c in df.columns]

def ols_residualize(y, X):
    """
    Retourne residus + intercept pour conserver le niveau moyen.
    y: Series
    X: DataFrame (peut contenir NaN, gérés par drop)
    """
    df = pd.concat([y, X], axis=1)
    df = df.dropna()
    if df.empty:
        return pd.Series(index=y.index, data=np.nan)
    y_clean = df.iloc[:,0]
    X_clean = df.iloc[:,1:]
    if X_clean.empty:
        return y
    Xc = sm.add_constant(X_clean, has_constant='add')
    try:
        model = sm.OLS(y_clean, Xc).fit()
        resid = model.resid + model.params.get('const', 0.0)
        out = pd.Series(index=y.index, data=np.nan)
        out.loc[resid.index] = resid
        return out
    except Exception:
        return y

def residualize_on_confond(df_subject_level, target_col, confond):
    cols = [c for c in confond if c in df_subject_level.columns]
    if not cols:
        return df_subject_level[target_col]
    X = df_subject_level[cols].copy()
    # Encodage catégoriel
    for c in cols:
        if X[c].dtype == 'object' or str(X[c].dtype).startswith('category'):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return ols_residualize(df_subject_level[target_col], X)

def prepare_long(df_bundle, metric_col, point_col):
    """
    Retourne DataFrame long: subject, point, value + colonnes meta (group, etc.).
    """
    needed = ['subject', point_col, metric_col]
    if not all(c in df_bundle.columns for c in needed):
        raise ValueError(f"Colonnes manquantes pour {metric_col}")
    meta_cols = [c for c in df_bundle.columns if c not in METRIC_COLUMNS_CANDIDATES]
    long_df = df_bundle[meta_cols + [metric_col]].rename(columns={metric_col:'value', point_col:'point'})
    return long_df

def pointwise_ttest(long_df, classif_col):
    """
    Retourne DataFrame point, p_raw, p_fdr, mean_g0, mean_g1, diff, sig_fdr.
    Suppose 2 groupes.
    """
    if classif_col not in long_df.columns:
        return pd.DataFrame()
    groups = long_df[classif_col].dropna().unique()
    if len(groups) != 2:
        return pd.DataFrame()
    g0, g1 = sorted(groups, key=lambda x: str(x))
    rows = []
    for pid, g in long_df.groupby('point'):
        sub = g[[classif_col,'value']].dropna()
        vals0 = sub.loc[sub[classif_col]==g0,'value'].astype(float)
        vals1 = sub.loc[sub[classif_col]==g1,'value'].astype(float)
        if len(vals0)>=2 and len(vals1)>=2:
            try:
                stat, p = ttest_ind(vals0, vals1, equal_var=False, nan_policy='omit')
            except Exception:
                p = np.nan
            rows.append({
                'point': pid,
                'p_raw': p,
                'mean_g0': vals0.mean(),
                'mean_g1': vals1.mean(),
                'diff': vals1.mean() - vals0.mean(),
                'n0': len(vals0),
                'n1': len(vals1)
            })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows).sort_values('point')
    res['p_raw'] = res['p_raw'].astype(float)
    # FDR
    valid_mask = res['p_raw'].notna()
    if valid_mask.any():
        reject, p_fdr, _, _ = multipletests(res.loc[valid_mask,'p_raw'], method='fdr_bh')
        res['p_fdr'] = np.nan
        res.loc[valid_mask,'p_fdr'] = p_fdr
        res['sig_fdr'] = res['p_fdr'] < 0.05
    else:
        res['p_fdr'] = np.nan
        res['sig_fdr'] = False
    return res

def group_test_afq(long_df, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM):
    """
    Test de différence de groupe avec correction multi-comparaisons (AFQ) pour obtenir alphaFWE.
    Un point est significatif si p_raw < alphaFWE.
    """
    if classif_col not in long_df.columns or long_df.empty:
        return pd.DataFrame()

    # 1. Retirer lignes sans label de groupe
    long_df = long_df[long_df[classif_col].notna()]
    if long_df.empty:
        return pd.DataFrame()

    groups = long_df[classif_col].dropna().unique()
    if len(groups) != 2:
        return pd.DataFrame()
    g0, g1 = sorted(groups, key=lambda x: str(x))

    # 2. Tableau sujet x point (après filtrage)
    pivot = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')

    # 3. Series des labels (index sujet)
    subj_groups = (long_df[['subject', classif_col]]
                   .drop_duplicates()
                   .set_index('subject')
                   .loc[pivot.index])  # aligner sur pivot

    # 4. Retirer sujets avec label NaN (sécurité)
    mask_valid_label = subj_groups[classif_col].notna()
    pivot = pivot.loc[mask_valid_label]
    subj_groups = subj_groups.loc[mask_valid_label]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    # 5. Retirer sujets avec NaN métriques (lignes contenant au moins un NaN)
    before_drop = pivot.shape[0]
    pivot = pivot.dropna(axis=0, how='any')
    if pivot.shape[0] < before_drop:
        subj_groups = subj_groups.loc[pivot.index]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    # 6. Encoder y
    y_series = subj_groups[classif_col].map({g0: 0, g1: 1})
    # Si mapping inattendu -> drop
    mask_y_valid = y_series.notna()
    pivot = pivot.loc[mask_y_valid]
    y_series = y_series.loc[mask_y_valid]
    if pivot.shape[0] < 4:
        return pd.DataFrame()

    y = y_series.values.astype(int)  # garanti sans NaN maintenant

    # 7. AFQ (permutations)
    try:
        alphaFWE, _, clusterFWE, _ = AFQ_MultiCompCorrection(pivot.values, y, alpha, nperm=int(nperm))
    except Exception as e:
        print(f"[group_test_afq] AFQ_MultiCompCorrection error: {e}")
        return pd.DataFrame()

    # 8. p-values Welch
    pvalues = []
    means0 = []
    means1 = []
    diffs = []
    n0_list = []
    n1_list = []
    for point in pivot.columns:
        vals0 = pivot.loc[y == 0, point].astype(float)
        vals1 = pivot.loc[y == 1, point].astype(float)
        if vals0.notna().sum() >= 2 and vals1.notna().sum() >= 2:
            try:
                t, p = sp_stats.ttest_ind(vals0, vals1, equal_var=False, nan_policy='omit')
            except Exception:
                p = np.nan
        else:
            p = np.nan
        pvalues.append(p)
        means0.append(vals0.mean())
        means1.append(vals1.mean())
        diffs.append(vals1.mean() - vals0.mean())
        n0_list.append(vals0.notna().sum())
        n1_list.append(vals1.notna().sum())

    pvalues = np.array(pvalues, dtype=float)
    sig_mask = (pvalues < alphaFWE)

    return pd.DataFrame({
        'point': pivot.columns,
        'p_raw': pvalues,
        'mean_g0': means0,
        'mean_g1': means1,
        'diff': diffs,
        'n0': n0_list,
        'n1': n1_list,
        'sig_afq': sig_mask,
        'alphaFWE': alphaFWE,
        'clusterFWE': clusterFWE
    })

def pointwise_correlation(long_df, var_col):
    """
    Corrélation (Pearson) + alphaFWE via permutations. Significatif si p_raw < alphaFWE.
    """
    if var_col not in long_df.columns:
        return pd.DataFrame()
    pivot_val = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    y = long_df[['subject', var_col]].drop_duplicates().set_index('subject').reindex(pivot_val.index)[var_col]
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y = y.astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        y = pd.to_numeric(y, errors='coerce')
    # Passage du Series directement (au lieu de y.values) pour conserver l’index
    res = _correlation_with_alphaFWE(pivot_val, y)
    return res['df']

def pointwise_partial_correlation(long_df, var_col, confond):
    """
    Corrélation partielle: résidualisation métrique & variable puis alphaFWE.
    """
    if var_col not in long_df.columns:
        return pd.DataFrame()
    pivot_val = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    subj_meta = long_df[['subject', var_col] + [c for c in confond if c in long_df.columns]].drop_duplicates().set_index('subject')
    pivot_val = pivot_val.reindex(subj_meta.index)
    var_res = residualize_on_confond(subj_meta.assign(target=subj_meta[var_col]), 'target', confond)

    X_res_cols = []
    for pid in pivot_val.columns:
        series_metric = pivot_val[pid]
        df_tmp = subj_meta.copy()
        df_tmp = df_tmp.assign(metric=series_metric)
        metric_resid = residualize_on_confond(df_tmp.assign(target=df_tmp['metric']), 'target', confond)
        X_res_cols.append(metric_resid)
    if not X_res_cols:
        return pd.DataFrame()
    X_res = pd.concat(X_res_cols, axis=1)
    X_res.columns = pivot_val.columns

    # Alignement explicite des sujets (évite mismatch longueur)
    common_index = X_res.index.intersection(var_res.index)
    X_res = X_res.loc[common_index]
    var_res_aligned = var_res.loc[common_index]

    res = _correlation_with_alphaFWE(X_res, var_res_aligned)
    return res['df']

# --- Nouvelle fonction utilitaire pour alphaFWE corrélations ---
def _correlation_with_alphaFWE(pivot_values: pd.DataFrame, y_vec,
                               nperm=1000, alpha=0.05):
    """
    Utilise AFQ_MultiCompCorrection pour obtenir alphaFWE & clusterFWE.
    y_vec peut être Series (préféré) ou array; on aligne sur l’intersection d’index.
    """
    if isinstance(y_vec, pd.Series):
        y_series = y_vec.copy()
    else:
        # Si array: construire un Series en supposant ordre identique
        y_series = pd.Series(y_vec, index=pivot_values.index[:len(y_vec)])

    # Intersection sujets
    common_index = pivot_values.index.intersection(y_series.index)
    pivot_values = pivot_values.loc[common_index]
    y_series = y_series.loc[common_index]

    # Si longueur insuffisante
    if len(y_series) < 4:
        return {'alphaFWE': np.nan, 'df': pd.DataFrame()}

    # Retirer lignes avec NaN dans y
    mask_y = y_series.notna()
    pivot_values = pivot_values.loc[mask_y]
    y_series = y_series.loc[mask_y]
    if len(y_series) < 4:
        return {'alphaFWE': np.nan, 'df': pd.DataFrame()}

    # Retirer sujets avec NaN sur au moins une colonne (cohérent avec analyse groupes)
    pivot_values = pivot_values.dropna(axis=0, how='any')
    y_series = y_series.reindex(pivot_values.index)
    if len(y_series) < 4:
        return {'alphaFWE': np.nan, 'df': pd.DataFrame()}

    try:
        alphaFWE, _, clusterFWE, _ = AFQ_MultiCompCorrection(pivot_values.values, y_series.values, alpha, nperm=int(nperm))
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        alphaFWE = np.nan
        clusterFWE = np.nan

    r_list = []
    p_list = []
    n_subj = len(y_series)
    for col in pivot_values.columns:
        xv = pivot_values[col].astype(float)
        mask = xv.notna() & y_series.notna()
        if mask.sum() >= 3:
            try:
                r, p = pearsonr(xv[mask], y_series[mask])
            except Exception:
                r, p = np.nan, np.nan
        else:
            r, p = np.nan, np.nan
        r_list.append(r)
        p_list.append(p)

    r_arr = np.array(r_list, dtype=float)
    p_arr = np.array(p_list, dtype=float)
    sig_mask = (p_arr < alphaFWE) if not np.isnan(alphaFWE) else np.zeros_like(p_arr, dtype=bool)

    df_out = pd.DataFrame({
        'point': pivot_values.columns,
        'r': r_arr,
        'p_raw': p_arr,
        'n': n_subj,
        'sig_afq': sig_mask,
        'alphaFWE': alphaFWE,
        'clusterFWE': clusterFWE
    })
    return {'alphaFWE': alphaFWE, 'df': df_out}

# def clean_missing_data(long_df, threshold=0.15):
#     """
#     Fake clean missing data function. Just keep the subjects of the list subject_list, and drop the others. Convert NaN to 0. set the first and last point to drop. Also only keep the bundles that are in bundle_to_keep.
#     """
#     subject_list = [
#     "01002", "01006", "01007", "01009", "01012", "01013", "01015", "01016", "01017", "01018", "01025", "01026", "01027", "01028", "01029", "01030", "01031", "01032", "01034", "01036", "01043", "01001", "01010", "01019", "01022", "01033", "01037", "01038", "01039", "01040", "01041", "01042",
#     "03002", "03003", "03004", "03005", "03008", "03009", "03010", "03021", "03022", "03024", "03011", "03012", "03013", "03014", "03015", "03016", "03017", "03018", "03019", "03020"
#     ]
#     if long_df.empty:
#         return long_df, [], []
#     subject_to_remove = sorted([s for s in long_df['subject'].unique() if s not in subject_list])
#     long_df_clean = long_df[~long_df['subject'].isin(subject_to_remove)].copy()
#     long_df_clean['value'] = long_df_clean['value'].fillna(0)
#     points_to_remove = [long_df_clean['point'].min(), long_df_clean['point'].max()]

    
#     return long_df_clean, subject_to_remove, []

def clean_missing_data(long_df, threshold=0.15):
    """
    Stratégie de gestion des NaN:
      - Si un point a des NaN pour >= threshold (proportion de sujets), on supprime ce point.
      - Sinon, on supprime les sujets qui ont des NaN (sur les points conservés).
    Retourne: long_df_clean, removed_subjects(list), removed_points(list)
    """
    if long_df.empty:
        return long_df, [], []
    # On considère uniquement les lignes (subject, point, value)
    pivot = long_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    n_subjects = pivot.shape[0]
    if n_subjects == 0:
        return long_df, [], []
    # Proportion de NaN par point
    prop_nan_point = pivot.isna().sum(axis=0) / n_subjects
    points_to_remove = sorted([p for p, prop in prop_nan_point.items() if prop >= threshold])

    #Ajouter points extrêmes si non déjà dans la liste
    min_point = pivot.columns.min()
    max_point = pivot.columns.max()
    if min_point not in points_to_remove:
        points_to_remove.insert(0, min_point)
    if max_point not in points_to_remove:
        points_to_remove.append(max_point)
    # Retirer ces points
    pivot_reduced = pivot.drop(columns=points_to_remove, errors='ignore')
    # Sujets à retirer: ayant encore des NaN
    subjects_to_remove = sorted(pivot_reduced.index[pivot_reduced.isna().any(axis=1)].tolist())
    # Filtrer long_df
    mask_points = ~long_df['point'].isin(points_to_remove)
    mask_subjects = ~long_df['subject'].isin(subjects_to_remove)
    long_df_clean = long_df[mask_points & mask_subjects].copy()
    return long_df_clean, subjects_to_remove, points_to_remove

def _annotate_missing(ax, removed_subjects, removed_points):
    """Ajoute un encart texte sur le graphique avec les infos de retrait."""
    lines = []
    if removed_subjects:
        ex = ", ".join(map(str, removed_subjects[:8]))
        if len(removed_subjects) > 8:
            ex += "..."
        lines.append(f"Sujets retirés ({len(removed_subjects)}): {ex}")
    if removed_points:
        exp = ", ".join(map(str, removed_points[:12]))
        if len(removed_points) > 12:
            exp += "..."
        lines.append(f"Points retirés ({len(removed_points)}) (≥15% NaN): {exp}")
    if lines:
        ax.text(0.01, 0.99, "\n".join(lines),
                ha='left', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, edgecolor='gray'))

def _export_group_csv(long_df, ttest_df, classif_col, metric_name, bundle_name, out_csv, analysis_label):
    """Construit et sauvegarde un CSV contenant les valeurs agrégées (mean, std, count) et p-values."""
    # Agrégations (mean, std, count) par point et groupe
    agg = (long_df.groupby(['point', classif_col])['value']
                  .agg(['mean', 'std', 'count'])
                  .reset_index())
    # Reshape pour colonnes par groupe
    pivot_mean = agg.pivot(index='point', columns=classif_col, values='mean')
    pivot_std = agg.pivot(index='point', columns=classif_col, values='std')
    pivot_n = agg.pivot(index='point', columns=classif_col, values='count')
    df_out = pd.DataFrame({'point': pivot_mean.index}).set_index('point')
    # Nommer colonnes
    for g in pivot_mean.columns:
        df_out[f'mean_{g}'] = pivot_mean[g]
        df_out[f'std_{g}'] = pivot_std[g]
        df_out[f'n_{g}'] = pivot_n[g]
    if ttest_df is not None and not ttest_df.empty:
        for col in ['p_raw','p_fdr','mean_g0','mean_g1','diff','n0','n1','sig_afq','sig_fdr','alphaFWE','clusterFWE']:
            if col in ttest_df.columns:
                df_out[col] = ttest_df.set_index('point')[col]
    df_out['bundle'] = bundle_name
    df_out['metric'] = metric_name
    df_out['classif'] = classif_col
    df_out['analysis'] = analysis_label
    df_out.reset_index().to_csv(out_csv, index=False)

def _annotate_pvalues(ax, ttest_df, sig_col, y_offset_factor=0.02, max_labels=30):
    """Ajoute des annotations de p-values au-dessus des points significatifs.
    Limite le nombre de labels pour éviter la surcharge graphique."""
    if ttest_df is None or ttest_df.empty:
        return
    sig_df = ttest_df[ttest_df[sig_col]] if sig_col in ttest_df.columns else pd.DataFrame()
    if sig_df.empty:
        return
    # Limiter le nombre de labels (priorité p les plus petites)
    if 'p_fdr' in sig_df.columns and sig_col != 'sig_afq':
        sig_df = sig_df.sort_values('p_fdr').head(max_labels)
        p_label_col = 'p_fdr'
    else:
        sig_df = sig_df.sort_values('p_raw').head(max_labels)
        p_label_col = 'p_raw'
    ylim = ax.get_ylim()
    y_span = ylim[1] - ylim[0]
    for _, row in sig_df.iterrows():
        pval = row.get(p_label_col, np.nan)
        if np.isnan(pval):
            continue
        # Position x = point, y légèrement au-dessus (ou en bas si manque de place)
        x = row['point']
        y = ylim[1] - y_span * 0.05  # 5% sous le max
        label = f"p={'{:.2e}'.format(pval) if pval < 0.001 else '{:.3f}'.format(pval)}"
        ax.text(x, y, label, rotation=90, ha='center', va='top', fontsize=7, color='red')

def plot_group_diff(long_df, ttest_df, classif_col, metric_name, bundle_name, out_path, title_suffix,
                    removed_subjects=None, removed_points=None, export_csv=True):
    """
    Ajout: deuxième axe Y avec barres des p-values (gris, alpha=0.3) + ligne horizontale rouge alphaFWE.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    sns.lineplot(
        data=long_df,
        x='point', y='value',
        hue=classif_col,
        estimator='mean',
        errorbar=('ci', 95)
    )
    sig_col = 'sig_afq' if (ttest_df is not None and not ttest_df.empty and 'sig_afq' in ttest_df.columns) else 'sig_fdr'
    if ttest_df is not None and not ttest_df.empty and ttest_df[sig_col].any():
        all_vals = pd.to_numeric(long_df['value'], errors='coerce')
        if all_vals.notna().any():
            low = np.nanquantile(all_vals, 0.02)
            high = np.nanquantile(all_vals, 0.98)
        else:
            low, high = 0, 1
        sig_line = ttest_df[['point']].copy()
        sig_line['sig_line'] = np.where(ttest_df[sig_col], high, low)
        label_txt = 'Significatif (alphaFWE)' if sig_col == 'sig_afq' else 'Significatif (FDR<0.05)'
        plt.plot(sig_line['point'], sig_line['sig_line'], color='red', linestyle='--', linewidth=2, label=label_txt)

    ax = plt.gca()

    # --- Nouveau: barres p-values + seuil ---
    if ttest_df is not None and not ttest_df.empty and 'p_raw' in ttest_df.columns:
        ax2 = ax.twinx()
        ax2.bar(ttest_df['point'], ttest_df['p_raw'],
                color='gray', alpha=0.1, width=1.0, label='p-value')
        if 'alphaFWE' in ttest_df.columns and not ttest_df['alphaFWE'].isna().all():
            alphaFWE_val = ttest_df['alphaFWE'].iloc[0]
            if not np.isnan(alphaFWE_val):
                ax2.axhline(alphaFWE_val, color='green', linestyle=':', linewidth=1.5, label=f'alphaFWE={alphaFWE_val:.3g}')
        ax2.set_ylabel("p-value")
        ymax = max(1.0, ttest_df['p_raw'].max() * 1.05)
        ax2.set_ylim(0, ymax)

        # Fusion des légendes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    plt.title(f"{bundle_name} - {metric_name} - {classif_col} {title_suffix}")
    plt.xlabel("Point")
    plt.ylabel(metric_name)
    _annotate_missing(ax, removed_subjects or [], removed_points or [])
    _annotate_pvalues(ax, ttest_df, sig_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    # Export CSV
    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + '.csv'
        analysis_lbl = 'raw' if 'raw' in out_path else 'corrected'
        _export_group_csv(long_df, ttest_df, classif_col, metric_name, bundle_name, out_csv, analysis_lbl)

def _export_corr_csv(corr_df, metric_name, bundle_name, var_name, out_csv, analysis_label):
    if corr_df is None or corr_df.empty:
        pd.DataFrame({'info':['aucune donnée']}).to_csv(out_csv, index=False)
        return
    df_out = corr_df.copy()
    # plus de p_fdr / sig_fdr -> alphaFWE & sig_afq
    df_out['bundle'] = bundle_name
    df_out['metric'] = metric_name
    df_out['variable'] = var_name
    df_out['analysis'] = analysis_label
    df_out.to_csv(out_csv, index=False)

def plot_correlation(corr_df, metric_name, bundle_name, var_name, out_path, title_suffix,
                     removed_subjects=None, removed_points=None, export_csv=True):
    """
    Ajout: barres p-values (gris 0.3) + ligne alphaFWE (rouge pointillée) + r.
    """
    plt.figure(figsize=(10,5))
    if corr_df.empty:
        plt.text(0.5,0.5,"Aucune donnée", ha='center')
        plt.title(f"{bundle_name} - {metric_name} - Corrélation {var_name} {title_suffix}")
    else:
        # Courbe r
        ax = plt.gca()
        ax.plot(corr_df['point'], corr_df['r'], color='tab:blue', label='r')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel("r de Pearson")
        # Axe p-values
        ax2 = ax.twinx()
        ax2.bar(corr_df['point'], corr_df['p_raw'], color='gray', alpha=0.3, width=1.0, label='p-value')
        if 'alphaFWE' in corr_df.columns and not corr_df['alphaFWE'].isna().all():
            alphaFWE_val = corr_df['alphaFWE'].iloc[0]
            if not np.isnan(alphaFWE_val):
                ax2.axhline(alphaFWE_val, color='red', linestyle=':', linewidth=1.5,
                            label=f'alphaFWE={alphaFWE_val:.3g}')
        ax2.set_ylabel("p-value")
        ax2.set_ylim(0, max(1.0, corr_df['p_raw'].max()*1.05))
        # Significatifs
        if 'sig_afq' in corr_df.columns and corr_df['sig_afq'].any():
            sig = corr_df[corr_df['sig_afq']]
            ax.scatter(sig['point'], sig['r'], color='red', s=25, label='p<alphaFWE')
            # Annotation p-values
            tmp = corr_df.copy()
            tmp['sig_afq'] = corr_df['sig_afq']
            _annotate_pvalues(ax2, tmp.rename(columns={'sig_afq':'sig_fdr'}), 'sig_fdr')  # réutilisation fonction (colonne renommée)
        h1,l1 = ax.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=8)
        plt.title(f"{bundle_name} - {metric_name} - Corrélation {var_name} {title_suffix}")
        _annotate_missing(ax, removed_subjects or [], removed_points or [])
    plt.xlabel("Point")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    if export_csv:
        out_csv = os.path.splitext(out_path)[0] + '.csv'
        analysis_lbl = 'raw' if 'raw' in out_path else 'partial'
        _export_corr_csv(corr_df, metric_name, bundle_name, var_name, out_csv, analysis_lbl)

# def _analyze_single_bundle(bundle_name, df, report_dir):
#     """
#     Analyse d'un bundle: tests de groupe avec AFQ + corrélations (FDR inchangé pour corrélations).
#     """
#     summary_rows = []
#     try:
#         point_col = detect_point_column(df)
#     except Exception:
#         return summary_rows
#     metric_cols = detect_metric_columns(df)
#     if not metric_cols:
#         return summary_rows

#     for metric_col in metric_cols:
#         long_df = prepare_long(df, metric_col, point_col)
#         long_df_clean, removed_subjects, removed_points = clean_missing_data(long_df, threshold=0.15)
#         if long_df_clean.empty:
#             continue

#         # Tests de groupe (AFQ)
#         for classif_col in classif_variables.keys():
#             if classif_col not in long_df_clean.columns:
#                 continue
#             # Brut
#             afq_raw = group_test_afq(long_df_clean, classif_col)
#             out_plot_raw = opj(report_dir, f"{bundle_name}_{metric_col}_{classif_col}_group_raw.png")
#             plot_group_diff(long_df_clean, afq_raw, classif_col, metric_col, bundle_name, out_plot_raw, "(brut)",
#                             removed_subjects, removed_points)

#             # Correction confonds: résidualiser par point puis refaire AFQ
#             conf_list = CLASSIF_CONFOUND_MAP.get(classif_col, confond_variables_with_control)
#             corrected_segments = []
#             for pid, g in long_df_clean.groupby('point'):
#                 subj_level = g[['subject','value'] + [c for c in conf_list if c in g.columns]].drop_duplicates(subset='subject').set_index('subject')
#                 y_res = residualize_on_confond(subj_level.assign(target=subj_level['value']), 'target', conf_list)
#                 g_corr = g.copy()
#                 g_corr['value'] = g_corr['subject'].map(y_res)
#                 corrected_segments.append(g_corr)
#             long_corr = pd.concat(corrected_segments, ignore_index=True)
#             afq_corr = group_test_afq(long_corr, classif_col)
#             out_plot_corr = opj(report_dir, f"{bundle_name}_{metric_col}_{classif_col}_group_corrected.png")
#             plot_group_diff(long_corr, afq_corr, classif_col, metric_col, bundle_name, out_plot_corr,
#                             "(corrigé confond)", removed_subjects, removed_points)

#             summary_rows.append({
#                 'bundle': bundle_name,
#                 'metric': metric_col,
#                 'type': f"group_{classif_col}",
#                 'n_sig_raw': int(afq_raw['sig_afq'].sum()) if not afq_raw.empty else 0,
#                 'n_sig_corrected': int(afq_corr['sig_afq'].sum()) if not afq_corr.empty else 0,
#                 'min_p_raw': float(afq_raw['p_raw'].min()) if not afq_raw.empty else np.nan,              # nouveau
#                 'min_p_corrected': float(afq_corr['p_raw'].min()) if not afq_corr.empty else np.nan,      # nouveau
#                 'max_abs_r_raw': np.nan,
#                 'max_abs_r_partial': np.nan,
#                 'removed_subjects': len(removed_subjects),
#                 'removed_points': len(removed_points),
#                 'alphaFWE_raw': float(afq_raw['alphaFWE'].iloc[0]) if not afq_raw.empty else np.nan,
#                 'alphaFWE_corrected': float(afq_corr['alphaFWE'].iloc[0]) if not afq_corr.empty else np.nan
#             })

#         # Corrélations (inchangé: FDR pointwise + partielle)
#         for var_col in corr_variables:
#             if var_col not in long_df_clean.columns:
#                 continue
#             corr_raw = pointwise_correlation(long_df_clean, var_col)
#             out_corr_raw = opj(report_dir, f"{bundle_name}_{metric_col}_{var_col}_corr_raw.png")
#             plot_correlation(corr_raw, metric_col, bundle_name, var_col, out_corr_raw, "(brut)",
#                              removed_subjects, removed_points)
#             conf_list = confond_variables_with_control
#             corr_partial = pointwise_partial_correlation(long_df_clean, var_col, conf_list)
#             out_corr_part = opj(report_dir, f"{bundle_name}_{metric_col}_{var_col}_corr_partial.png")
#             plot_correlation(corr_partial, metric_col, bundle_name, var_col, out_corr_part,
#                              "(corrigé confond)", removed_subjects, removed_points)
#             summary_rows.append({
#                 'bundle': bundle_name,
#                 'metric': metric_col,
#                 'type': f"corr_{var_col}",
#                 'n_sig_raw': int(corr_raw['sig_fdr'].sum()) if not corr_raw.empty else 0,
#                 'n_sig_corrected': int(corr_partial['sig_fdr'].sum()) if not corr_partial.empty else 0,
#                 'max_abs_effect_raw': np.nan,
#                 'max_abs_effect_corr': np.nan,
#                 'max_abs_r_raw': float(corr_raw['r'].abs().max()) if not corr_raw.empty else np.nan,
#                 'max_abs_r_partial': float(corr_partial['r'].abs().max()) if not corr_partial.empty else np.nan,
#                 'removed_subjects': len(removed_subjects),
#                 'removed_points': len(removed_points),
#                 'alphaFWE_raw': np.nan,
#                 'alphaFWE_corrected': np.nan
#             })

#     return summary_rows

# --- Helpers parallélisation ---

def _prepare_group_bundle(bundle_name, df, metric_col, classif_col):
    """
    Prépare les structures raw / corrigé pour un bundle donné (une métrique / une classif).
    Retourne dict ou None si non exploitable.
    """
    if classif_col not in df.columns or metric_col not in df.columns:
        return None
    try:
        point_col = detect_point_column(df)
    except Exception:
        return None
    long_df = prepare_long(df, metric_col, point_col)
    long_df_clean, removed_subjects, removed_points = clean_missing_data(long_df, threshold=0.15)
    if long_df_clean.empty:
        return None
    conf_list = CLASSIF_CONFOUND_MAP.get(classif_col, confond_variables_with_control)
    corrected_segments = []
    for pid, g in long_df_clean.groupby('point'):
        subj_level = g[['subject','value'] + [c for c in conf_list if c in g.columns]].drop_duplicates(subset='subject').set_index('subject')
        y_res = residualize_on_confond(subj_level.assign(target=subj_level['value']), 'target', conf_list)
        g_corr = g.copy()
        g_corr['value'] = g_corr['subject'].map(y_res)
        corrected_segments.append(g_corr)
    long_corr = pd.concat(corrected_segments, ignore_index=True)
    pivot_raw = long_df_clean.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    pivot_corr = long_corr.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
    return {
        'bundle': bundle_name,
        'metric': metric_col,
        'classif': classif_col,
        'long_raw': long_df_clean,
        'long_corr': long_corr,
        'pivot_raw': pivot_raw,
        'pivot_corr': pivot_corr,
        'removed_subjects': removed_subjects,
        'removed_points': removed_points
    }

# --- Nouveau: worker par variable de classification ---
def _worker_classif_variable(classif_col, bundle_data, metric_set, report_dir,
                             AFQ_NPERM=5000, AFQ_ALPHA=0.05, MULTI_BUNDLE_AFQ=False):
    """
    Version simplifiée:
      - RAW: group_test_afq
      - CORR: résidualisation point-par-point puis group_test_afq
    """
    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    summary_rows = []
    if not bundle_data:
        return summary_rows

    for metric_col in tqdm(metric_set, desc=f"classif:{classif_col}"):
        for bundle_name, df in tqdm(bundle_data.items(), desc=f"{classif_col}-{metric_col}", leave=False):
            try:
                point_col = detect_point_column(df)
            except Exception:
                continue
            if metric_col not in df.columns or classif_col not in df.columns:
                continue

            # Long format
            try:
                long_raw = prepare_long(df, metric_col, point_col)
            except Exception:
                continue

            # Filtrer sujets sans label de groupe avant nettoyage (évite NaN plus tard)
            long_raw = long_raw[long_raw[classif_col].notna()]
            if long_raw.empty:
                continue

            # Nettoyage (points / sujets NaN métriques)
            long_raw_clean, removed_subjects, removed_points = clean_missing_data(long_raw, threshold=0.15)
            if long_raw_clean.empty:
                continue

            # Analyse RAW
            raw_res = group_test_afq(long_raw_clean, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM)
            if not raw_res.empty:
                out_plot_raw = opj(figure_dir, f"{bundle_name}_{metric_col}_{classif_col}_group_raw.png")
                plot_group_diff(long_raw_clean, raw_res, classif_col, metric_col, bundle_name, out_plot_raw, "(brut)",
                                removed_subjects, removed_points)

            # Résidualisation CORR
            conf_list = CLASSIF_CONFOUND_MAP.get(classif_col, confond_variables_with_control)
            if conf_list:
                corrected_segments = []
                for pid, gpt in long_raw_clean.groupby('point'):
                    subj_level = (gpt[['subject', 'value'] + [c for c in conf_list if c in gpt.columns]]
                                  .drop_duplicates(subset='subject')
                                  .set_index('subject'))
                    y_res = residualize_on_confond(subj_level.assign(target=subj_level['value']),
                                                   'target', conf_list)
                    g_corr = gpt.copy()
                    g_corr['value'] = g_corr['subject'].map(y_res)
                    corrected_segments.append(g_corr)
                long_corr = pd.concat(corrected_segments, ignore_index=True) if corrected_segments else pd.DataFrame()
            else:
                long_corr = long_raw_clean.copy()

            corr_res = group_test_afq(long_corr, classif_col, alpha=AFQ_ALPHA, nperm=AFQ_NPERM) if not long_corr.empty else pd.DataFrame()
            if not corr_res.empty:
                out_plot_corr = opj(figure_dir, f"{bundle_name}_{metric_col}_{classif_col}_group_corrected.png")
                plot_group_diff(long_corr, corr_res, classif_col, metric_col, bundle_name, out_plot_corr,
                                "(corrigé confond)", removed_subjects, removed_points)

            summary_rows.append({
                'bundle': bundle_name,
                'metric': metric_col,
                'type': f"group_{classif_col}",
                'n_sig_raw': int(raw_res['sig_afq'].sum()) if not raw_res.empty else 0,
                'n_sig_corrected': int(corr_res['sig_afq'].sum()) if not corr_res.empty else 0,
                'min_p_raw': float(raw_res['p_raw'].min()) if not raw_res.empty else np.nan,
                'min_p_corrected': float(corr_res['p_raw'].min()) if not corr_res.empty else np.nan,
                'max_abs_r_raw': np.nan,
                'max_abs_r_partial': np.nan,
                'removed_subjects': len(removed_subjects),
                'removed_points': len(removed_points),
                'alphaFWE_raw': float(raw_res['alphaFWE'].iloc[0]) if not raw_res.empty else np.nan,
                'alphaFWE_corrected': float(corr_res['alphaFWE'].iloc[0]) if not corr_res.empty else np.nan
            })
    return summary_rows

# --- Nouveau: worker par variable de corrélation ---
def _worker_corr_variable(var_col, bundle_data, metric_set, report_dir):
    rows = []
    if not bundle_data:
        return rows
    figure_dir = opj(report_dir, "figures")
    ensure_dir(figure_dir)
    for bundle_name, df in tqdm(bundle_data.items(), desc=f"corr:{var_col} bundles"):
        try:
            point_col = detect_point_column(df)
        except Exception:
            continue
        metric_cols = [m for m in metric_set if m in df.columns]
        for metric_col in tqdm(metric_cols, desc=f"{var_col}-{bundle_name} métriques", leave=False):
            long_df = prepare_long(df, metric_col, point_col)
            long_df_clean, removed_subjects, removed_points = clean_missing_data(long_df, threshold=0.15)
            if long_df_clean.empty or var_col not in long_df_clean.columns:
                continue
            corr_raw = pointwise_correlation(long_df_clean, var_col)
            out_corr_raw = opj(figure_dir, f"{bundle_name}_{metric_col}_{var_col}_corr_raw.png")
            plot_correlation(corr_raw, metric_col, bundle_name, var_col, out_corr_raw, "(brut)",
                             removed_subjects, removed_points)
            corr_partial = pointwise_partial_correlation(long_df_clean, var_col, confond_variables_with_control)
            out_corr_part = opj(figure_dir, f"{bundle_name}_{metric_col}_{var_col}_corr_partial.png")
            plot_correlation(corr_partial, metric_col, bundle_name, var_col, out_corr_part,
                             "(corrigé confond)", removed_subjects, removed_points)
            rows.append({
                'bundle': bundle_name,
                'metric': metric_col,
                'type': f"corr_{var_col}",
                'n_sig_raw': int(corr_raw['sig_afq'].sum()) if not corr_raw.empty else 0,
                'n_sig_corrected': int(corr_partial['sig_afq'].sum()) if not corr_partial.empty else 0,
                'min_p_raw': float(corr_raw['p_raw'].min()) if not corr_raw.empty else np.nan,
                'min_p_corrected': float(corr_partial['p_raw'].min()) if not corr_partial.empty else np.nan,
                'max_abs_r_raw': float(corr_raw['r'].abs().max()) if not corr_raw.empty else np.nan,
                'max_abs_r_partial': float(corr_partial['r'].abs().max()) if not corr_partial.empty else np.nan,
                'removed_subjects': len(removed_subjects),
                'removed_points': len(removed_points),
                'alphaFWE_raw': float(corr_raw['alphaFWE'].iloc[0]) if not corr_raw.empty else np.nan,
                'alphaFWE_corrected': float(corr_partial['alphaFWE'].iloc[0]) if not corr_partial.empty else np.nan
            })
    return rows

def generate_report(bundle_names, csv_files, report_dir="report_output", n_jobs=1):
    """
    Génère le rapport complet.
    Parallélisation désormais par variable (classification ou corrélation) : chaque worker traite
    une variable sur l'ensemble des bundles et métriques.
    """
    ensure_dir(report_dir)
    bundle_data = {}
    # Chargement bundles
    # (on suppose bundle_names déjà préparé avant l'appel)
    for bundle_name in tqdm([bn.replace("_","") for bn in bundle_names], desc="Chargement des bundles"):
        bundle_csvs = [f for f in csv_files if f.get_entities()['bundle']==bundle_name]
        if not bundle_csvs:
            continue
        df, _ = load_and_merge_bundle_csvs(bundle_name, bundle_csvs)
        bundle_data[bundle_name] = df
    if not bundle_data:
        print("Aucun bundle à analyser.")
        return

    # Déterminer toutes les métriques présentes
    metric_set = set()
    for df in bundle_data.values():
        metric_set.update(detect_metric_columns(df))
    metric_set = sorted(metric_set)

    # Liste des jobs variables
    var_jobs = [('classif', c) for c in classif_variables.keys()] + [('corr', c) for c in corr_variables]

    summary_rows = []
    if n_jobs > 1 and var_jobs:
        from joblib import Parallel, delayed
        def _dispatch(job):
            kind, var = job
            if kind == 'classif':
                return _worker_classif_variable(var, bundle_data, metric_set, report_dir)
            else:
                return _worker_corr_variable(var, bundle_data, metric_set, report_dir)
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_dispatch)(job) for job in var_jobs
        )
        for r in results:
            summary_rows.extend(r)
    else:
        for kind, var in tqdm(var_jobs, desc="Variables"):
            if kind == 'classif':
                summary_rows.extend(_worker_classif_variable(var, bundle_data, metric_set, report_dir))
            else:
                summary_rows.extend(_worker_corr_variable(var, bundle_data, metric_set, report_dir))

    if not summary_rows:
        print("Aucune donnée pour le rapport.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = opj(report_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)

    md_lines = ["# Rapport Tractométrie",
                "",
                f"Généré sur {pd.Timestamp.now()}",
                "",
                "## Résumé (table)",
                "",
                summary_df.to_markdown(index=False),
                "",
                "## Bundles (analyse de groupe) zones significatives (AFQ)"]
    group_sig = summary_df[summary_df['type'].str.startswith('group_')]
    sig_any = group_sig[(group_sig['n_sig_raw']>0)|(group_sig['n_sig_corrected']>0)]
    if sig_any.empty:
        md_lines.append("- Aucun")
    else:
        for _, r in sig_any.iterrows():
            md_lines.append(f"- {r.bundle} / {r.metric} / {r.type}: n_sig (brut={r.n_sig_raw}, corrigé={r.n_sig_corrected}) alphaFWE(brut={r.alphaFWE_raw}, corr={r.alphaFWE_corrected})")
    md_lines.append("")
    md_lines.append("## Bundles avec corrélations significatives (AFQ)")  # renommé (plus FDR)
    corr_sig = summary_df[summary_df['type'].str.startswith('corr_')]
    corr_any = corr_sig[(corr_sig['n_sig_raw']>0)|(corr_sig['n_sig_corrected']>0)]
    if corr_any.empty:
        md_lines.append("- Aucun")
    else:
        for _, r in corr_any.iterrows():
            md_lines.append(f"- {r.bundle} / {r.metric} / {r.type}: points sig (brut={r.n_sig_raw}, partiel={r.n_sig_corrected}) max|r| (brut={r.max_abs_r_raw:.3f}, partiel={r.max_abs_r_partial:.3f})")

    report_md = opj(report_dir, "report.md")
    with open(report_md, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Rapport généré: {report_md}")
    print(f"Résumé CSV: {summary_csv}")

    # Documentation statistique détaillée
    stat_doc_lines = [
        "# Documentation statistique",
        "",
        "Ce document décrit précisément les traitements et tests statistiques effectués dans `generate_report_v2.py`.",
        "",
        "## 1. Préparation des données",
    "Pour chaque bundle et chaque métrique scalaire tractométrique (FA, MD, RD, AD, IFW, IRF si présentes):",
    "- Fusion des CSV sujets + métadonnées cliniques/démographiques + actimétrie. Conversion en format long: colonnes `subject`, `point`, `value` (métrique) + variables explicatives.",
        "",
        "### Gestion des données manquantes",
        "1. Construction d'un tableau sujets x points.",
        "2. Suppression des points avec proportion de valeurs manquantes >= 15%.",
        "3. Suppression des sujets contenant encore des NaN sur les points conservés.",
        "Ces exclusions sont annotées sur chaque figure.",
        "",
        "## 2. Comparaisons de groupes",
        "Pour chaque variable de classification (ex: `group`, `apathy`):",
        "- Test t de Welch point-par-point -> p_raw.",
        "- AFQ fournit un seuil global alphaFWE (permutations) contrôlant l'erreur familiale.",
        "- Nouveau critère: sig_afq = (p_raw < alphaFWE) directement (plus de get_significant_areas).",
        "- Deux analyses: (i) brute; (ii) corrigée des confondants.",
        "",
        "### Correction des confondants",
        "Pour l'analyse corrigée: pour chaque point, on résidualise la valeur métrique individuelle par rapport aux confondants définis (âge, sexe, ville ± variables cliniques selon le contexte).",
        "Modèle OLS: value ~ conf1 + conf2 + ...; on conserve les résidus + intercept pour préserver le niveau moyen.",
        "Les résidus remplacent les valeurs d'origine avant de relancer l'analyse AFQ.",
        "",
        "## 3. Corrélations (variables quantitatives ou ordinales)",
        "Deux versions par variable: brute et partielle (corrigée des confondants).",
        "- Corrélation de Pearson point-par-point: r, p_raw.",
        "- Contrôle FWE (alphaFWE) via AFQ_MultiCompCorrection (permutations).",
        "- Significativité: p_raw < alphaFWE -> sig_afq.",
        "- Corrélation partielle: résidualisation (métrique par point + variable) sur confondants, même procédure AFQ.",
        "",
        "## 4. Résidualisation",
        "Encodage des variables catégorielles en codes entiers (puisque OLS). Valeurs manquantes ignorées via suppression des lignes incomplètes dans le modèle de régression.",
        "La fonction `ols_residualize` ajoute l'intercept aux résidus afin que la distribution conserve son niveau moyen approximatif.",
        "",
        "## 5. Gestion des p-values et annotations graphiques",
        "- Groupes: barres p-values + ligne alphaFWE; p_raw < alphaFWE signifie sig_afq.",
        "- Corrélations: idem (barres p-values, ligne alphaFWE).",
        "",
        "## 6. Limitations",
        "- Les tests supposent indépendance approximative des sujets et distribution raisonnablement gaussienne des métriques.",
        "- Les p-values annotées sont limitées pour lisibilité et peuvent cacher d'autres points significatifs si >30.",
        "- La résidualisation point-par-point peut introduire un léger bruit si beaucoup de NaN variables confondantes.",
    ]
    stat_doc_path = opj(report_dir, 'statistics_methodology.md')
    with open(stat_doc_path, 'w') as f:
        f.write('\n'.join(stat_doc_lines))
    print(f"Documentation statistique: {stat_doc_path}")

# Intégrer dans le flux existant, après le chargement des bundles
# Remplacer la boucle finale simple par accumulation + génération de rapport
bundles_loaded = {}
for bundle_name in bundle_names:
    bundle_csvs = [f for f in csv_files if f.get_entities()['bundle']==bundle_name]
    if not bundle_csvs:
        continue
    bundle_df, bundle_metric_files_dict = load_and_merge_bundle_csvs(bundle_name, bundle_csvs)
    bundles_loaded[bundle_name] = bundle_df

pipeline="no_actimetry"
#get hostname
hostname = os.uname().nodename
print(f"Hostname: {hostname}")
n_jobs=16
output_dir = f'/home/ndecaux/report_{pipeline}_tractseg'

if hostname=='calcarine':
    pipeline="actimetry"
    corr_variables=actimetry_columns
    classif_variables={}
    output_dir = f'/home/ndecaux/NAS_EMPENN/share/projects/actidep/report_{pipeline}_calcarine'
    n_jobs=80
os.makedirs(output_dir, exist_ok=True)
print(f"Rapport dans {output_dir}")
# Lancer l’analyse complête (adapter pour parallélisation)
generate_report(bundles_loaded.keys(), csv_files, report_dir=output_dir, n_jobs=n_jobs)

