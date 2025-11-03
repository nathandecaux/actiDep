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
import statsmodels.api as sm  # NEW: régression OLS pour correction par nuisances
from sklearn.linear_model import LinearRegression  # NEW: pour correction type unconfound
from tractseg.libs.AFQ_MultiCompCorrection import AFQ_MultiCompCorrection, get_significant_areas  # NEW: AFQ
from tractseg.libs import metric_utils  # NEW: pour unconfound

class SimpleBundleViewer(param.Parameterized):
    """
    Visualiseur simple pour afficher les courbes de métriques par sujet 
    pour un faisceau et une métrique donnés, avec hover pour afficher l'ID du sujet.
    """
    # Paramètres interactifs
    metric = param.Selector(default='FA', objects=['FA', 'MD', 'RD', 'AD','IFW','IRF'], doc="Métrique à visualiser")
    selected_bundle = param.Selector(objects=[], doc="Faisceau sélectionné")
    selected_subjects = param.ListSelector(default=[], objects=[], doc="Sujets à afficher (tous si vide)")
    group_by_column = param.Selector(default=None, objects=[None], doc="Colonne pour grouper les sujets")
    show_individual_curves = param.Boolean(default=True, doc="Afficher les courbes individuelles")
    show_group_averages = param.Boolean(default=False, doc="Afficher les moyennes de groupe")
    filter_by_subjects_file = param.Boolean(default=False, doc="Filtrer par subjects.txt")
    # Ajout: options de rééchantillonnage
    resample_curves = param.Boolean(default=False, doc="Rééchantillonner les courbes")
    resample_points = param.Integer(default=100, bounds=(2, 2000), doc="Nombre de points après rééchantillonnage")
    # Nouveau: choix de la méthode d'interpolation
    interpolation_method = param.Selector(default='linear', objects=['linear', 'nearest', 'cubic'], doc="Méthode d'interpolation")

    enable_nuisance_correction = param.Boolean(default=False, doc="Corriger la métrique par régression sur des nuisances")
    nuisance_columns = param.ListSelector(default=[], objects=[], doc="Colonnes de nuisance à régresser")
    nuisance_correction_method = param.Selector(
        default='unconfound',
        objects=['unconfound', 'ols'],
        doc="Méthode de correction des nuisances (unconfound multi-cible ou OLS point-par-point)"
    )

    enable_correlation_plot = param.Boolean(default=False, doc="Afficher la corrélation point-par-point")
    correlation_column = param.Selector(default=None, objects=[None], doc="Colonne pour la corrélation")
    correlation_method = param.Selector(default='pearson', objects=['pearson', 'spearman'], doc="Méthode de corrélation")

    selected_point = param.Integer(default=None, allow_None=True, doc="Point sélectionné sur la courbe de corrélation")

    # NEW: paramètres d’affichage de la significativité (grouping)
    show_group_significance = param.Boolean(default=False, doc="Mettre en avant les points significatifs (grouping)")
    significance_alpha = param.Number(default=0.05, bounds=(0.0001, 0.5), doc="Seuil alpha de significativité")
    multiple_correction = param.Selector(default='none', objects=['none', 'bonferroni', 'fdr_bh'], doc="Correction multi-tests")

    # NEW: options AFQ pour la significativité
    fwe_method = param.Selector(default='alphaFWE', objects=['alphaFWE', 'clusterFWE'], doc="Méthode AFQ pour la significativité")
    nperm = param.Integer(default=2000, bounds=(100, 100000), doc="Nombre de permutations AFQ")
    # Constantes (peuvent être des paramètres si besoin de flexibilité)
    # model = 'staniz'
    # pipeline = 'tractometry'

    model = 'MCM'
    pipeline = 'hcp_association_tractseg'
    def __init__(self, dataset_path: str = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids", **params):
        super().__init__(**params)
        self.dataset_path = dataset_path
        self.subjects_file_path = "/home/ndecaux/Code/actiDep/subjects.txt"
        # NEW: init précoce avant tout watcher
        self._nuisance_cache = {}
        self._tap_stream = None
        self.valid_subjects_from_file = set()
        self.df_all_bundles_for_metric = None # DataFrame pour la métrique actuelle
        self.bundle_names_for_metric = []   # Noms des faisceaux pour la métrique actuelle
        self.all_subjects = []  # Liste de tous les sujets disponibles
        self.participants_info = None  # DataFrame avec les informations supplémentaires des sujets
        
        # Charger la liste des sujets valides depuis subjects.txt
        self._load_subjects_from_file()
        
        # Charger les informations des participants
        self._load_participants_info()
        
        # Charger les données pour la métrique par défaut et initialiser les sélecteurs
        self._load_data_for_current_metric()

    def _load_subjects_from_file(self):
        """Charge la liste des sujets depuis le fichier subjects.txt"""
        try:
            with open(self.subjects_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Ignorer les commentaires et les lignes vides
                    if line and not line.startswith('#') and not line.startswith('subject_id'):
                        parts = line.split()
                        if len(parts) >= 1:
                            subject_id = parts[0].replace('sub-', '')  # Enlever le préfixe sub- si présent
                            self.valid_subjects_from_file.add(subject_id)

            print(f"Sujets trouvés dans {self.subjects_file_path}: {len(self.valid_subjects_from_file)} sujets")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {self.subjects_file_path}: {e}")
            self.valid_subjects_from_file = set()

    def _filter_subjects_by_file(self, subjects_list):
        """Filtre la liste des sujets selon ceux présents dans subjects.txt"""
        if not self.filter_by_subjects_file or not self.valid_subjects_from_file:
            return subjects_list
        
        filtered_subjects = [s for s in subjects_list if s in self.valid_subjects_from_file]
        print(f"Filtrage activé: {len(filtered_subjects)}/{len(subjects_list)} sujets conservés")
        return filtered_subjects

    def _load_participants_info(self):
        """Charge les informations supplémentaires des participants depuis le fichier Excel."""
        participants_file = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx"
        
        try:
            self.participants_info = pd.read_excel(participants_file)
            print(f"Informations des participants chargées: {len(self.participants_info)} lignes")
            print(f"Colonnes disponibles: {list(self.participants_info.columns)}")
            
            # Standardiser la colonne participant_id si nécessaire
            if 'participant_id' in self.participants_info.columns:
                # Enlever le préfixe 'sub-' s'il existe pour correspondre aux IDs des sujets
                self.participants_info['subject_id'] = self.participants_info['participant_id'].str.replace('sub-', '', regex=False)
            elif 'subject_id' not in self.participants_info.columns:
                print("Attention: Aucune colonne 'participant_id' ou 'subject_id' trouvée dans les données des participants")
                self.participants_info = None
                return
            
            # Mettre à jour les options pour le groupement (exclure les colonnes ID)
            grouping_columns = [col for col in self.participants_info.columns 
                              if col not in ['participant_id', 'subject_id'] and 
                              self.participants_info[col].notna().sum() > 0]
            
            self.param.group_by_column.objects = [None] + grouping_columns
            # NEW: proposer les mêmes colonnes pour nuisances et corrélation
            self.param.nuisance_columns.objects = grouping_columns
            self.param.correlation_column.objects = [None] + grouping_columns
        except FileNotFoundError:
            print(f"Fichier participants non trouvé: {participants_file}")
            self.participants_info = None
        except Exception as e:
            print(f"Erreur lors du chargement des informations des participants: {e}")
            self.participants_info = None

    @param.depends('metric', 'filter_by_subjects_file', watch=True)
    def _trigger_data_reload_and_update_bundles(self):
        """
        Appelé lorsque la métrique ou le filtrage change. Recharge les données et met à jour 
        la liste des faisceaux et le faisceau sélectionné.
        """
        print(f"Changement de métrique vers: {self.metric} ou filtrage: {self.filter_by_subjects_file}. Rechargement des données...")
        self._load_data_for_current_metric()

    # NEW: utilitaire centralisé
    def _clear_nuisance_cache(self, reason: str = ""):
        if not hasattr(self, "_nuisance_cache"):
            self._nuisance_cache = {}
        self._nuisance_cache.clear()
        if reason:
            print(f"Cache nuisance vidé ({reason}).")
        else:
            print("Cache nuisance vidé.")

    @param.depends('enable_nuisance_correction', 'nuisance_columns', 'metric',
                   'nuisance_correction_method', watch=True)
    def _invalidate_nuisance_cache(self):
        """NEW: Invalide le cache quand la config de correction change."""
        self._clear_nuisance_cache("configuration")

    # NEW: invalidation supplémentaire sur changement de faisceau / grouping / filtrage
    @param.depends('selected_bundle', 'group_by_column', 'filter_by_subjects_file', watch=True)
    def _invalidate_nuisance_cache_selection(self):
        self._clear_nuisance_cache("sélection / bundle / filtrage")

    def _load_data_for_current_metric(self):
        """
        Charge les données CSV pour la métrique actuellement sélectionnée (`self.metric`).
        Met à jour `self.df_all_bundles_for_metric`, `self.bundle_names_for_metric`,
        et les options/valeur de `self.selected_bundle`.
        """
        print(f"Chargement des données pour la métrique: {self.metric}...")
        
        pattern = opj(self.dataset_path, 'derivatives', self.pipeline, 'sub-*', 'metric', 
                     f'*_metric-{self.metric}*_model-{self.model}*_mean.csv')
        
        csv_files = glob.glob(pattern)
        print(f"Trouvé {len(csv_files)} fichiers CSV pour la métrique {self.metric}")
        
        all_data_list = []
        current_bundle_names_from_files = [] 
        
        if not csv_files:
            print(f"Aucun fichier de métriques trouvé pour {self.metric} avec le pattern {pattern}")
        else:
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Regex pour extraire l'ID du sujet (peut inclure des lettres et chiffres)
                subject_id_match = re.search(r'sub-([a-zA-Z0-9]+)', filename)
                if subject_id_match:
                    subject_id = subject_id_match.group(1)
                else:
                    print(f"ID du sujet non trouvé dans {filename}. Fichier ignoré.")
                    continue
                
                # Filtrer par subjects.txt si activé
                if self.filter_by_subjects_file and self.valid_subjects_from_file:
                    if subject_id not in self.valid_subjects_from_file:
                        continue
                    
                try:
                    # Essayer différents séparateurs
                    try:
                        df_subj = pd.read_csv(csv_file, sep=';')
            
                    except:
                        df_subj = pd.read_csv(csv_file, sep=',')
                    
                    if df_subj.empty:
                        print(f"Fichier vide: {csv_file}")
                        continue

                    # Récupérer les noms des faisceaux à partir du premier fichier valide
                    if not current_bundle_names_from_files: # Premier fichier valide trouvé
                        current_bundle_names_from_files = df_subj.columns.tolist()
                    
                    for bundle_name_col in current_bundle_names_from_files:
                        if bundle_name_col in df_subj.columns:
                            values = df_subj[bundle_name_col].values
                            for point_idx, value in enumerate(values):
                                all_data_list.append({
                                    'subject': subject_id, 
                                    'bundle': bundle_name_col,
                                    'point': point_idx,
                                    'value': value
                                })
                except (pd.errors.EmptyDataError, FileNotFoundError, ValueError) as e:
                    print(f"Erreur lors du chargement ou traitement de {csv_file}: {e}")
                    continue
        
        if not all_data_list:
            self.df_all_bundles_for_metric = pd.DataFrame(columns=['subject', 'bundle', 'point', 'value'])
            print(f"Aucune donnée n'a pu être extraite pour la métrique {self.metric}.")
            self.all_subjects = []
        else:
            self.df_all_bundles_for_metric = pd.DataFrame(all_data_list)
            # Mettre à jour la liste de tous les sujets disponibles
            raw_subjects = sorted(list(self.df_all_bundles_for_metric['subject'].unique()))
            self.all_subjects = self._filter_subjects_by_file(raw_subjects)
        
        # Ajouter les informations des participants si disponibles
        if self.participants_info is not None and not self.df_all_bundles_for_metric.empty:
            self.df_all_bundles_for_metric = self.df_all_bundles_for_metric.merge(
                self.participants_info, 
                left_on='subject', 
                right_on='subject_id', 
                how='left'
            )
            print(f"Informations des participants ajoutées aux données métriques")
        
        # Mettre à jour les noms des faisceaux uniques pour la métrique actuelle
        if not self.df_all_bundles_for_metric.empty:
            self.bundle_names_for_metric = sorted(list(self.df_all_bundles_for_metric['bundle'].unique()))
        else:
            self.bundle_names_for_metric = []
        
        print(f"Données chargées pour {self.metric}: {len(self.df_all_bundles_for_metric)} lignes, {len(self.bundle_names_for_metric)} faisceaux uniques.")

        # Mettre à jour le paramètre selected_bundle (options et valeur)
        old_selected_bundle = self.selected_bundle
        self.param.selected_bundle.objects = self.bundle_names_for_metric
        
        if self.bundle_names_for_metric:
            if old_selected_bundle in self.bundle_names_for_metric:
                self.selected_bundle = old_selected_bundle # Conserver la sélection si elle est toujours valide
            else:
                self.selected_bundle = self.bundle_names_for_metric[0] # Sinon, prendre le premier de la liste
        else:
            self.selected_bundle = None # Aucun faisceau disponible

        # Mettre à jour les options pour le sélecteur de sujets
        old_selected_subjects = self.selected_subjects
        self.param.selected_subjects.objects = self.all_subjects
        
        # Conserver les sélections précédentes si elles sont toujours valides
        if self.all_subjects:
            if old_selected_subjects:
                # Garder seulement les sujets qui existent encore dans les données actuelles
                self.selected_subjects = [s for s in old_selected_subjects if s in self.all_subjects]
            else:
                # Par défaut, sélectionner tous les sujets (en les laissant vide, tous seront affichés)
                self.selected_subjects = []
        else:
            self.selected_subjects = []
        # NEW: purge cache nuisance car nouvelles données chargées
        self._clear_nuisance_cache("rechargement des données")

    # Ajout: utilitaire de rééchantillonnage par sujet
    def _resample_bundle_df(self, bundle_df: pd.DataFrame) -> pd.DataFrame:
        if not self.resample_curves or self.resample_points is None or self.resample_points < 2:
            return bundle_df

        method = getattr(self, 'interpolation_method', 'linear')
        resampled = []
        cols_to_keep = [c for c in bundle_df.columns if c not in ['point', 'value']]

        for subject_id, g in bundle_df.groupby('subject'):
            g = g.sort_values('point')
            x = g['point'].to_numpy(dtype=float)
            y = g['value'].to_numpy(dtype=float)
            if len(x) == 0:
                continue

            # Assurer des x strictement croissants et uniques pour les interpolateurs
            xu, idx_unique = np.unique(x, return_index=True)
            yu = y[idx_unique]

            new_x = np.linspace(float(xu[0]), float(xu[-1]) if len(xu) > 1 else float(xu[0]), self.resample_points)

            if len(xu) == 1:
                new_y = np.full_like(new_x, yu[0], dtype=float)
            else:
                if method == 'nearest':
                    # Interpolation au plus proche voisin (vectorisée)
                    idxs = np.searchsorted(xu, new_x, side='left')
                    idxs = np.clip(idxs, 0, len(xu) - 1)
                    left_idx = np.clip(idxs - 1, 0, len(xu) - 1)
                    right_idx = idxs
                    choose_left = np.abs(new_x - xu[left_idx]) <= np.abs(xu[right_idx] - new_x)
                    nearest_idx = np.where(choose_left, left_idx, right_idx)
                    new_y = yu[nearest_idx]
                elif method == 'cubic':
                    try:
                        from scipy.interpolate import interp1d
                        f = interp1d(xu, yu, kind='cubic', bounds_error=False, fill_value='extrapolate')
                        new_y = f(new_x)
                    except Exception as e:
                        print(f"Interpolation 'cubic' indisponible, utilisation de 'linear' (raison: {e})")
                        new_y = np.interp(new_x, xu, yu)
                else:
                    # 'linear' par défaut
                    new_y = np.interp(new_x, xu, yu)

            base_vals = {col: g.iloc[0][col] for col in cols_to_keep}
            resampled.append(pd.DataFrame({
                **base_vals,
                'point': new_x,
                'value': new_y
            }))

        if not resampled:
            return bundle_df
        return pd.concat(resampled, ignore_index=True)

    # NEW: correction par régression linéaire des variables de nuisance (point-par-point)
    def _apply_nuisance_correction_ols(self, bundle_df: pd.DataFrame, nuis_cols, preserve_group: bool) -> pd.DataFrame:
        """
        Correction OLS point-par-point (value ~ nuisances).
        Si preserve_group=True et group_by_column binaire: on n'enlève pas l'effet de groupe (exclusion de la régression).
        """
        df = bundle_df.copy()
        group_col = self.group_by_column if preserve_group and self.group_by_column in df.columns else None

        # Préparer encodage nuisances
        enc = {}
        for c in nuis_cols:
            if c == group_col:
                continue
            s = df[c]
            if s.dtype == 'object' or str(s.dtype).startswith('category'):
                enc[c] = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
            else:
                enc[c] = pd.to_numeric(s, errors='coerce')
        if not enc:
            return df
        enc_df = pd.DataFrame(enc, index=df.index)

        corrected_vals = {}
        for pid, idx in df.groupby('point').groups.items():
            sub_idx = pd.Index(idx)
            y = pd.to_numeric(df.loc[sub_idx, 'value'], errors='coerce')
            X = enc_df.loc[sub_idx]

            # Filtrage colonnes valides
            valid_cols = [c for c in X.columns if X[c].notna().sum() >= 3 and X[c].nunique(dropna=True) > 1]
            if not valid_cols:
                continue
            Xi = X[valid_cols]
            # Imputation simple
            Xi = Xi.apply(lambda col: col.fillna(col.mean()), axis=0)

            try:
                Xi_const = sm.add_constant(Xi, has_constant='add')
                model = sm.OLS(y, Xi_const, missing='drop').fit()
                resid = model.resid
                const_val = float(model.params.get('const', 0.0))
                y_corr = resid + const_val
                corrected_vals[(pid,)] = (y_corr.index, y_corr.values)
            except Exception:
                continue

        if not corrected_vals:
            return df

        # Appliquer corrections
        for (pid,), (indices, vals) in corrected_vals.items():
            df.loc[indices, 'value'] = vals
        return df

    # NEW: méthode unconfound multi-cibles (régression simultanée points)
    def _apply_nuisance_correction_unconfound(self, bundle_df: pd.DataFrame, nuis_cols, preserve_group: bool) -> pd.DataFrame:
        if not nuis_cols:
            return bundle_df
        df = bundle_df.copy()
        group_col = self.group_by_column if preserve_group and self.group_by_column in df.columns else None

        # Construire matrice sujets x points
        wide = df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
        if wide.empty:
            return df

        # Construire DataFrame des nuisances par sujet
        meta = (df[['subject'] + nuis_cols].drop_duplicates()
                  .set_index('subject').reindex(wide.index))
        # Encodage numérique
        Xcols = []
        for c in nuis_cols:
            if c == group_col:
                continue
            s = meta[c]
            if s.dtype == 'object' or str(s.dtype).startswith('category'):
                Xcols.append(s.astype('category').cat.codes.replace(-1, np.nan).astype(float))
            else:
                Xcols.append(pd.to_numeric(s, errors='coerce'))
        if not Xcols:
            return df
        X = pd.concat(Xcols, axis=1)
        X.columns = [c for c in nuis_cols if c != group_col]

        # Supprimer colonnes constantes / quasi vides
        keep = [c for c in X.columns if X[c].notna().sum() >= 3 and X[c].nunique(dropna=True) > 1]
        if not keep:
            return df
        X = X[keep]
        # Imputation simple
        X = X.apply(lambda col: col.fillna(col.mean()), axis=0)

        # Ajout constante
        X_const = np.column_stack([np.ones(len(X)), X.values])

        Y = wide.values  # (n_subjects, n_points)
        mask_subject = ~np.isnan(Y).all(axis=1)
        Xc = X_const[mask_subject]
        Yc = Y[mask_subject]

        # Régression linéaire multi-sorties point par point (résidus + intercept)
        try:
            # Centrer Y pour stabilité
            Ymean = np.nanmean(Yc, axis=0, keepdims=True)
            Yc_centered = np.where(np.isnan(Yc), Ymean, Yc) - Ymean
            # Pseudo-inverse
            beta = np.linalg.pinv(Xc) @ Yc_centered
            fitted = Xc @ beta
            resid = Yc_centered - fitted
            corrected = resid + Ymean  # restaurer niveau moyen
            # Remettre dans matrice complète
            Y_corr = Y.copy()
            Y_corr[mask_subject] = corrected
            # Recomposer long
            wide_corr = pd.DataFrame(Y_corr, index=wide.index, columns=wide.columns)
            df_corr = (wide_corr.stack()
                       .rename('value')
                       .reset_index())
            df_corr.rename(columns={'level_1': 'point'}, inplace=True)
            # Fusionner variables annexes
            keep_cols = [c for c in df.columns if c not in ['value']]
            df_out = df_corr.merge(df[keep_cols].drop_duplicates(), on=['subject', 'point'], how='left')
            return df_out
        except Exception:
            # Fallback silencieux
            return df

    # NEW: méthode unifiée avec cache
    def _apply_nuisance_correction(self, bundle_df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_nuisance_cache"):
            self._nuisance_cache = {}
        if not self.enable_nuisance_correction or not self.nuisance_columns:
            # Toujours renvoyer une copie pour éviter effets de référence
            return bundle_df.copy(deep=True)
        nuis_cols = [c for c in self.nuisance_columns if c in bundle_df.columns]
        if not nuis_cols:
            return bundle_df.copy(deep=True)
        preserve_group = True
        subjects_key = tuple(sorted(bundle_df['subject'].unique()))
        points_key = len(bundle_df['point'].unique())
        key = (self.metric, self.selected_bundle, subjects_key, points_key,
               tuple(sorted(nuis_cols)), self.nuisance_correction_method,
               self.group_by_column, preserve_group)
        cached = self._nuisance_cache.get(key)
        if cached is not None:
            print(f"[Nuisance] Utilisation du cache (key hash={hash(key) % 10**8})")
            return cached.copy(deep=True)
        print(f"[Nuisance] Calcul (méthode={self.nuisance_correction_method}) (key hash={hash(key) % 10**8})")
        if self.nuisance_correction_method == 'ols':
            corrected = self._apply_nuisance_correction_ols(bundle_df, nuis_cols, preserve_group)
        else:
            corrected = self._apply_nuisance_correction_unconfound(bundle_df, nuis_cols, preserve_group)
        # Stocker copie profonde
        self._nuisance_cache[key] = corrected.copy(deep=True)
        return corrected.copy(deep=True)

    # NEW: calcul de corrélation point-par-point pour une colonne donnée
    def _compute_pointwise_correlation(self, bundle_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule r (et p) pour chaque 'point' entre 'value' et la colonne sélectionnée.
        Gère automatiquement l'encodage numérique des colonnes catégorielles.
        """
        col = self.correlation_column
        method = self.correlation_method
        if not col or col not in bundle_df.columns:
            return pd.DataFrame(columns=['point', 'r', 'p', 'n'])

        # Encoder la colonne si catégorielle
        s = bundle_df[col]
        if s.dtype == 'object' or str(s.dtype).startswith('category'):
            x_full = s.astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            x_full = pd.to_numeric(s, errors='coerce')

        # Import optionnel de scipy.stats
        try:
            from scipy.stats import pearsonr, spearmanr
        except Exception:
            pearsonr = spearmanr = None

        rows = []
        for pid, idx in bundle_df.groupby('point').groups.items():
            idx = pd.Index(idx)
            y = pd.to_numeric(bundle_df.loc[idx, 'value'], errors='coerce')
            x = x_full.loc[idx]

            # Supprimer les NaN alignés
            mask = x.notna() & y.notna()
            xv = x[mask].values
            yv = y[mask].values
            n = int(mask.sum())
            if n < 3:
                rows.append({'point': pid, 'r': np.nan, 'p': np.nan, 'n': n})
                continue

            r = np.nan
            p = np.nan
            try:
                if method == 'spearman' and spearmanr is not None:
                    res = spearmanr(xv, yv, nan_policy='omit')
                    r, p = float(res.correlation), float(res.pvalue)
                elif method == 'pearson' and pearsonr is not None:
                    r, p = pearsonr(xv, yv)
                    r, p = float(r), float(p)
                else:
                    # Fallback sans scipy: corrcoef pour r, p inconnu
                    r = float(np.corrcoef(xv, yv)[0, 1])
                    p = np.nan
            except Exception:
                r, p = np.nan, np.nan

            rows.append({'point': pid, 'r': r, 'p': p, 'n': n})

        return pd.DataFrame(rows).sort_values('point')

    # NEW: plot détaillé pour un point cliqué (scatter + régression + labels)
    @param.depends('selected_point', 'enable_correlation_plot', 'correlation_column',
                   'selected_bundle', 'metric', 'selected_subjects',
                   'enable_nuisance_correction', 'nuisance_columns')
    def _point_detail_plot(self):
        if not self.enable_correlation_plot:
            return hv.Text(0.5, 0.5, "Activez le mode corrélation.")
        if not self.correlation_column:
            return hv.Text(0.5, 0.5, "Choisissez une colonne de corrélation.")
        if self.selected_point is None:
            return hv.Text(0.5, 0.5, "Cliquez un point pour afficher les détails.")

        # Récup données filtrées et corrigées
        if self.df_all_bundles_for_metric is None or self.df_all_bundles_for_metric.empty or not self.selected_bundle:
            return hv.Text(0.5, 0.5, "Aucune donnée.")
        df = self.df_all_bundles_for_metric[self.df_all_bundles_for_metric['bundle'] == self.selected_bundle]
        if self.selected_subjects:
            df = df[df['subject'].isin(self.selected_subjects)]
        if df.empty:
            return hv.Text(0.5, 0.5, "Aucune donnée pour ce faisceau/sélection.")
        df = self._apply_nuisance_correction(df)
        dfp = df[df['point'] == self.selected_point]
        # NEW: si aucun match exact (flottants), choisir le point le plus proche
        if dfp.empty:
            try:
                pts = df['point'].astype(float).to_numpy()
                nearest = float(pts[np.argmin(np.abs(pts - float(self.selected_point)))])
                dfp = df[np.isclose(df['point'].astype(float), nearest)]
            except Exception:
                pass
        if dfp.empty or self.correlation_column not in dfp.columns:
            return hv.Text(0.5, 0.5, "Données indisponibles pour ce point.")

        xcol = self.correlation_column
        ycol = 'value'
        # Scatter
        scatter = hv.Scatter(dfp, kdims=xcol, vdims=[ycol, 'subject']).opts(
            size=8, color='navy', alpha=0.8, tools=['hover'],
            # FIX: tooltip correct sur la métrique
            hover_tooltips=[('Sujet', '@subject'), (xcol, f'@{xcol}'), (self.metric, f'@{ycol}{{0.3f}}')]
        )

        overlays = [scatter]

        # Labels sujets au-dessus de chaque point
        y_vals = pd.to_numeric(dfp[ycol], errors='coerce')
        if y_vals.notna().any():
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            dy = (y_max - y_min) * 0.03 if y_max > y_min else max(abs(y_max) * 0.03, 1e-6)
            labels_df = pd.DataFrame({
                'x': dfp[xcol],
                'y': y_vals + dy,
                'text': dfp['subject'].astype(str)
            })
            labels = hv.Labels(labels_df, kdims=['x', 'y'], vdims=['text']).opts(text_color='black', text_alpha=0.9)
            overlays.append(labels)

        # Régression linéaire si x numérique
        x_num = pd.to_numeric(dfp[xcol], errors='coerce')
        mask = x_num.notna() & y_vals.notna()
        if mask.sum() >= 2:
            X = sm.add_constant(x_num[mask].astype(float), has_constant='add')
            y = y_vals[mask].astype(float)
            try:
                m = sm.OLS(y, X).fit()
                x_line = np.linspace(float(x_num[mask].min()), float(x_num[mask].max()), 100)
                Xl = sm.add_constant(x_line, has_constant='add')
                y_line = m.predict(Xl)
                reg = hv.Curve((x_line, y_line), kdims=xcol, vdims=ycol).opts(color='firebrick', line_width=2)
                overlays.append(reg)
            except Exception:
                pass  # ignore si régression échoue

        title = f"Détails point {self.selected_point} — {self.metric} vs {xcol}"
        return hv.Overlay(overlays).opts(
            title=title, xlabel=xcol, ylabel=self.metric, width=450, height=600, show_legend=False
        )

    # NEW: calcul des p-values point-par-point pour le grouping via AFQ
    def _compute_group_significance(self, bundle_df: pd.DataFrame):
        """
        Utilise AFQ_MultiCompCorrection pour obtenir alphaFWE/clusterFWE, puis calcule les p-values
        par t-test (2 groupes) point-par-point et renvoie un masque binaire de significativité.
        """
        col = self.group_by_column
        if not col or col not in bundle_df.columns:
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        # Identifier 2 groupes (comme plot_tractometry_results)
        grp_vals = bundle_df[col].dropna()
        uniq = list(pd.Series(sorted(grp_vals.unique(), key=lambda x: str(x))))
        if len(uniq) != 2:
            # AFQ group testing nécessite 2 groupes; sinon on ne calcule pas
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        # Préparer la matrice [subjects x points] alignée et le vecteur y (0/1)
        # Filtrer sujets disponibles
        subjects = sorted(bundle_df['subject'].dropna().unique())
        # Pivot: lignes = sujets, colonnes = points, valeurs = metric
        pivot = bundle_df.pivot_table(index='subject', columns='point', values='value', aggfunc='mean')
        # Interpolation pour combler des éventuels trous (par ligne)
        pivot = pivot.apply(lambda r: r.astype(float).interpolate(limit_direction='both'), axis=1)
        # Garder seulement les sujets avec au moins 2 valeurs
        valid_rows = pivot.notna().sum(axis=1) >= 2
        pivot = pivot.loc[valid_rows]
        if pivot.empty:
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        # Construire y selon l’ordre des sujets
        subj_meta = (bundle_df[['subject', col]]
                     .drop_duplicates()
                     .set_index('subject')
                     .reindex(pivot.index))
        # Mappe 2 groupes en 0/1
        g0, g1 = uniq[0], uniq[1]
        y = subj_meta[col].map({g0: 0, g1: 1}).astype(float)
        valid_y = y.notna()
        pivot = pivot.loc[valid_y]
        y = y.loc[valid_y]
        if pivot.shape[0] < 3:
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        # AFQ: alpha corrigé
        values_allp = pivot.to_numpy()  # [subjects, NR_POINTS]
        try:
            alphaFWE, statFWE, clusterFWE, _ = AFQ_MultiCompCorrection(values_allp, y.values,
                                                                        float(self.significance_alpha),
                                                                        nperm=int(self.nperm))
        except Exception:
            # AFQ indisponible: ne rien tracer
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        # p-values par point via t-test (Group0 vs Group1)
        from scipy.stats import ttest_ind
        grp0_idx = (y.values == 0)
        grp1_idx = (y.values == 1)
        if grp0_idx.sum() < 2 or grp1_idx.sum() < 2:
            return pd.DataFrame(columns=['point', 'pvalue', 'sig'])

        vals0 = values_allp[grp0_idx, :]
        vals1 = values_allp[grp1_idx, :]
        # t-test indépendant avec égalité de variances relâchée
        _, pvals = ttest_ind(vals0, vals1, equal_var=False, nan_policy='omit')
        pvals = np.asarray(pvals, dtype=float)

        # Masque binaire de significativité selon la méthode AFQ choisie
        if self.fwe_method == 'alphaFWE':
            sig = pvals < float(alphaFWE)
        else:
            # clusterFWE: utiliser get_significant_areas comme dans plot_tractometry_results
            sig = get_significant_areas(pvals, clusterFWE, float(self.significance_alpha)).astype(bool)

        return pd.DataFrame({
            'point': pivot.columns.astype(float).values,
            'pvalue': pvals,
            'sig': sig
        })

    @param.depends('selected_bundle', 'metric', 'selected_subjects', 'group_by_column',
                   'show_individual_curves', 'show_group_averages',
                   'resample_curves', 'resample_points', 'interpolation_method',
                   'enable_nuisance_correction', 'nuisance_columns',
                   'enable_correlation_plot', 'correlation_column', 'correlation_method',
                   'show_group_significance', 'significance_alpha', 'multiple_correction',
                   # NEW: recalcul AFQ
                   'fwe_method', 'nperm')
    def plot_subject_curves(self):
        """
        Crée un graphique HoloViews affichant les courbes de chaque sujet pour 
        le faisceau et la métrique sélectionnés, avec possibilité de groupement.
        """
        if self.df_all_bundles_for_metric is None or self.df_all_bundles_for_metric.empty:
            return hv.Text(0.5, 0.5, f"Aucune donnée chargée pour la métrique {self.metric}.")
        
        if not self.selected_bundle:
            return hv.Text(0.5, 0.5, "Veuillez sélectionner un faisceau dans la liste.")

        # Filtrer les données pour le faisceau sélectionné
        bundle_df = self.df_all_bundles_for_metric[
            self.df_all_bundles_for_metric['bundle'] == self.selected_bundle
        ]

        if bundle_df.empty:
            return hv.Text(0.5, 0.5, f"Aucune donnée pour le faisceau '{self.selected_bundle}' avec la métrique '{self.metric}'.")

        # Filtrer pour les sujets sélectionnés (si spécifiés)
        if self.selected_subjects:
            bundle_df = bundle_df[bundle_df['subject'].isin(self.selected_subjects)]
            if bundle_df.empty:
                return hv.Text(0.5, 0.5, "Aucun des sujets sélectionnés n'a de données pour ce faisceau.")

        # 1) Correction par nuisances puis 2) Rééchantillonnage
        bundle_df = self._apply_nuisance_correction(bundle_df)
        # 2) Rééchantillonnage si demandé
        if self.resample_curves:
            bundle_df = self._resample_bundle_df(bundle_df)
        # 3) Branche corrélation
        if self.enable_correlation_plot:
            if not self.correlation_column:
                return hv.Text(0.5, 0.5, "Veuillez sélectionner une colonne pour la corrélation.")

            zero = hv.HLine(0).opts(color='gray', line_dash='dashed', line_width=1)
            overlays = []

            if self.group_by_column and self.group_by_column in bundle_df.columns:
                colors = ['firebrick', 'steelblue', 'seagreen', 'orange', 'purple', 'brown', 'pink', 'gray']
                for i, (gname, gdf) in enumerate(bundle_df.groupby(self.group_by_column)):
                    corr_df = self._compute_pointwise_correlation(gdf)
                    if corr_df.empty or corr_df['r'].isna().all():
                        continue
                    curve = hv.Curve(corr_df, kdims='point', vdims=['r', 'p', 'n'], label=str(gname)).opts(
                        line_width=3, color=colors[i % len(colors)],
                        tools=['hover', 'tap'],
                        hover_tooltips=[('Point', '@point'), ('r', '@r{0.3f}'), ('p', '@p{0.3g}'), ('n', '@n')]
                    )
                    overlays.append(curve)
                if not overlays:
                    return hv.Text(0.5, 0.5, "Corrélation indisponible pour la sélection et les groupes.")
                overlay = hv.Overlay(overlays)
                title = f"Corrélation ({self.correlation_method}) {self.metric} ~ {self.correlation_column} - {self.selected_bundle} (par {self.group_by_column})"
            else:
                corr_df = self._compute_pointwise_correlation(bundle_df)
                if corr_df.empty or corr_df['r'].isna().all():
                    return hv.Text(0.5, 0.5, "Corrélation indisponible pour la sélection actuelle.")
                overlay = hv.Curve(corr_df, kdims='point', vdims=['r', 'p', 'n'], label='r').opts(
                    line_width=3, color='firebrick',
                    tools=['hover', 'tap'],
                    hover_tooltips=[('Point', '@point'), ('r', '@r{0.3f}'), ('p', '@p{0.3g}'), ('n', '@n')]
                )
                title = f"Corrélation ({self.correlation_method}) {self.metric} ~ {self.correlation_column} - {self.selected_bundle}"

            # NEW: Tap stream réutilisable (évite accumulation)
            if self._tap_stream is None:
                self._tap_stream = hv.streams.Tap(source=overlay)
                def _on_tap(x, y):
                    if x is None:
                        return
                    try:
                        pts = np.asarray(sorted(bundle_df['point'].astype(float).unique()))
                        nearest = float(pts[np.argmin(np.abs(pts - float(x)))])
                        self.selected_point = int(round(nearest)) if np.isclose(nearest, round(nearest)) else nearest
                    except Exception:
                        pass
                self._tap_stream.add_subscriber(_on_tap)
            else:
                self._tap_stream.source = overlay
            return (overlay * zero).opts(
                title=title,
                xlabel="Position le long du faisceau",
                ylabel="Corrélation (r)",
                width=900, height=600,
                show_legend=True if isinstance(overlay, hv.Overlay) else False,
                legend_position='top_right'
            )

        # 4) Tracé standard (pas de corrélation) — bundle_df est déjà rééchantillonné si demandé
        overlays = []
        
        # Afficher les courbes individuelles si demandé
        if self.show_individual_curves:
            for subject_id, subject_data in bundle_df.groupby('subject'):
                curve = hv.Curve(subject_data, kdims='point', vdims=['value', 'subject'], 
                                label=str(subject_id)).opts(
                                    line_width=1.5,
                                    alpha=0.3,
                                    tools=['hover'],
                                    hover_line_color='red',
                                    hover_tooltips=[('Sujet', '@subject'), ('Position', '@point'), ('Valeur', '@value')]
                                )
                overlays.append(curve)
        
        # Afficher les moyennes de groupe si demandé et si une colonne de groupement est sélectionnée
        if self.show_group_averages and self.group_by_column:
            if self.group_by_column in bundle_df.columns:
                # Grouper par la colonne sélectionnée et calculer les statistiques
                group_stats = bundle_df.groupby([self.group_by_column, 'point']).agg({
                    'value': ['mean', 'std', 'count']
                }).reset_index()
                
                # Aplatir les colonnes
                group_stats.columns = [self.group_by_column, 'point', 'mean', 'std', 'count']
                
                # Remplacer les NaN par 0 pour l'écart-type
                group_stats['std'] = group_stats['std'].fillna(0)
                
                # Couleurs pour les groupes
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                
                for i, (group_name, group_data) in enumerate(group_stats.groupby(self.group_by_column)):
                    color = colors[i % len(colors)]
                    
                    # Ligne de moyenne
                    mean_curve = hv.Curve(group_data, kdims='point', vdims='mean', 
                                        label=f'{group_name} (n={group_data["count"].iloc[0]})').opts(
                                            line_width=3,
                                            color=color,
                                            alpha=0.8
                                        )
                    
                    # Zone d'écart-type (mean ± std)
                    upper = group_data['mean'] + group_data['std']
                    lower = group_data['mean'] - group_data['std']
                    
                    # Créer les données pour la zone
                    area_data = pd.DataFrame({
                        'point': np.concatenate([group_data['point'].values, group_data['point'].values[::-1]]),
                        'value': np.concatenate([upper.values, lower.values[::-1]])
                    })
                    
                    area = hv.Area(area_data, kdims='point', vdims='value').opts(
                        fill_color=color,
                        fill_alpha=0.2,
                        line_alpha=0
                    )
                    
                    overlays.extend([area, mean_curve])
        
        # NEW: overlay des points significatifs via AFQ quand grouping est sélectionné
        if self.show_group_significance and self.group_by_column and self.group_by_column in bundle_df.columns:
            sig_df = self._compute_group_significance(bundle_df)
            if not sig_df.empty and sig_df['sig'].any():
                # Ligne binaire aux quantiles (style tractseg)
                y = pd.to_numeric(bundle_df['value'], errors='coerce')
                if y.notna().any():
                    low = float(np.nanquantile(y, 0.02))
                    high = float(np.nanquantile(y, 0.98))
                else:
                    low, high = 0.0, 1.0
                sig_line = np.where(sig_df['sig'].values, high, low)
                sig_overlay_df = pd.DataFrame({'point': sig_df['point'].values, 'sig_line': sig_line})
                sig_curve = hv.Curve(sig_overlay_df, kdims='point', vdims='sig_line').opts(
                    color='red', line_dash='dashed', line_width=2, alpha=0.9, tools=[]
                )
                overlays.append(sig_curve)

        if not overlays:
            return hv.Text(0.5, 0.5, f"Aucune courbe à afficher pour {self.selected_bundle}.")

        title_parts = [f"{self.metric} pour le faisceau {self.selected_bundle}"]
        if self.group_by_column and self.show_group_averages:
            title_parts.append(f"(groupé par {self.group_by_column})")

        overlay = hv.Overlay(overlays).opts(
            title=" ".join(title_parts),
            xlabel="Position le long du faisceau",
            ylabel=self.metric,
            width=900,
            height=600,
            tools=['hover'],
            show_legend=True,
            legend_position='top_right'
        )
        
        return overlay

    def view(self):
        """Crée la vue Panel complète avec les contrôles et le graphique."""
        
        # Widgets de contrôle pour les paramètres
        controls = pn.Param(
            self.param,
            parameters=[
                'metric', 'selected_bundle', 'group_by_column',
                'show_individual_curves', 'show_group_averages', 'filter_by_subjects_file',
                'resample_curves', 'resample_points', 'interpolation_method',
                'enable_nuisance_correction', 'nuisance_columns', 'nuisance_correction_method',
                'enable_correlation_plot', 'correlation_column', 'correlation_method',
                # NEW: contrôles de significativité
                'show_group_significance', 'significance_alpha', 'multiple_correction',
                # NEW: AFQ
                'fwe_method', 'nperm'
            ],
            widgets={
                'metric': {'type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 'options': ['FA', 'MD', 'RD', 'AD','IFW','IRF']},
                'selected_bundle': pn.widgets.Select,
                'group_by_column': pn.widgets.Select,
                'show_individual_curves': pn.widgets.Checkbox,
                'show_group_averages': pn.widgets.Checkbox,
                'filter_by_subjects_file': pn.widgets.Checkbox,
                'resample_curves': pn.widgets.Checkbox,
                'resample_points': {'type': pn.widgets.IntInput, 'start': 2, 'end': 2000, 'step': 1},
                'interpolation_method': pn.widgets.Select,
                'enable_nuisance_correction': pn.widgets.Checkbox,
                'nuisance_columns': {'type': pn.widgets.MultiSelect, 'size': 6},
                # NEW: widgets de corrélation
                'enable_correlation_plot': pn.widgets.Checkbox,
                'correlation_column': pn.widgets.Select,
                'correlation_method': pn.widgets.RadioButtonGroup,
                # NEW: widgets de significativité
                'show_group_significance': pn.widgets.Checkbox,
                'significance_alpha': {'type': pn.widgets.FloatInput, 'start': 0.0001, 'end': 0.5, 'step': 0.005},
                'multiple_correction': pn.widgets.Select,
                # NEW: widgets AFQ
                'fwe_method': pn.widgets.Select,
                'nperm': {'type': pn.widgets.IntInput, 'start': 100, 'end': 100000, 'step': 100},
            },
            width=300
        )
        
        # Widget spécifique pour la sélection multiple des sujets
        subjects_widget = pn.widgets.MultiSelect(
            name="Sujets à afficher (vide = tous)",
            options=self.all_subjects,
            value=self.selected_subjects,
            height=200,
            width=280
        )
        
        # Lier le widget de sélection des sujets au paramètre correspondant
        subjects_widget.link(self, value='selected_subjects')
        
        # Le graphique HoloViews, qui se met à jour dynamiquement
        plot_pane = pn.pane.HoloViews(self.plot_subject_curves, sizing_mode="stretch_both")
        # NEW: panneau de détails pour le point sélectionné
        details_pane = pn.pane.HoloViews(self._point_detail_plot, sizing_mode="stretch_both")

        # Information sur les données chargées
        info_text = pn.pane.Markdown("## Informations")
        if self.participants_info is not None:
            info_text.object += f"\n- Données des participants chargées: {len(self.participants_info)} sujets"
            info_text.object += f"\n- Colonnes disponibles pour groupement: {len(self.param.group_by_column.objects)-1}"
        
        if self.valid_subjects_from_file:
            info_text.object += f"\n- Sujets dans subjects.txt: {len(self.valid_subjects_from_file)}"
        
        # Utilisation d'un template pour une mise en page agréable
        template = pn.template.FastListTemplate(
            title="Visualiseur de Faisceaux",
            sidebar=[
                pn.pane.Markdown("## Options"),
                controls,
                pn.pane.Markdown("## Sélection des sujets"),
                subjects_widget,
                info_text
            ],
            # NEW: place le plot principal et le détail côte à côte
            main=[pn.Row(plot_pane, details_pane)],
            header_background='#1f77b4',
            sidebar_width=320,
        )
        
        # Fonction pour mettre à jour le titre du template dynamiquement
        @pn.depends(self.param.metric, self.param.selected_bundle, self.param.selected_subjects, watch=True)
        def _update_template_title(metric_val, bundle_val, subjects_val):
            if bundle_val:
                n_subjects = len(subjects_val) if subjects_val else len(self.all_subjects)
                template.title = f"Visualiseur: {metric_val} pour {bundle_val} ({n_subjects} sujets)"
            else:
                template.title = f"Visualiseur: {metric_val} (aucun faisceau sélectionné)"
        
        # Appel initial pour définir le titre
        _update_template_title(self.metric, self.selected_bundle, self.selected_subjects)
        
        return template


def run_simple_viewer_app():
    """Fonction principale pour lancer le visualiseur simple."""
    
    # Vous pouvez changer le chemin par défaut ici si nécessaire, ou le passer en argument
    # Par exemple: viewer = SimpleBundleViewer(dataset_path="/autre/chemin/bids")
    # db_root = '/home/ndecaux/Data/dysdiago/'
    db_root = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
    viewer = SimpleBundleViewer(db_root) 
    
    try:
        app_view = viewer.view()
        
        print("Visualiseur simple prêt. Démarrage du serveur Panel...")
        # Utiliser un port différent si le port 5007 est déjà utilisé
        app_view.show(open=True, title="Visualiseur de Faisceaux ActiDep") 
        
    except (IOError, KeyError, IndexError) as e:
        print(f"Erreur lors du lancement du visualiseur simple: {e}")
        import traceback
        traceback.print_exc()

def run_interactive():
    """Fonction pour lancer le visualiseur en mode interactif."""
    run_simple_viewer_app()

def main():
    """
    Fonction principale pour générer un dashboard statique.
    Retourne True si le dashboard a été généré avec succès, False sinon.
    """
    try:
        viewer = SimpleBundleViewer()
        app_view = viewer.view()
        
        # Générer un fichier HTML statique
        output_file = "tractometry_dashboard.html"
        app_view.save(output_file)
        print(f"Dashboard enregistré dans {output_file}")
        return True
        
    except (IOError, KeyError, IndexError) as e:
        print(f"Erreur lors de la génération du dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_simple_viewer_app()

