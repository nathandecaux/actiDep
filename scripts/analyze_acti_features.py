# simple_apaty_experiments.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_selection import SelectFpr, VarianceThreshold, RFE
from scipy.stats import ttest_ind, pearsonr, spearmanr
from joblib import Parallel, delayed
import statsmodels.api as sm
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def save_ttest_results_to_csv(ttest_report_12h, ttest_report_3d, res_dir, task_name):
    """Sauvegarde les résultats du t-test pour chaque feature en CSV."""
    all_ttest_results = []
    
    # Ajouter les résultats 12h
    for result in ttest_report_12h:
        result_copy = result.copy()
        result_copy['feature_type'] = '12h'
        result_copy['task'] = task_name
        all_ttest_results.append(result_copy)
    
    # Ajouter les résultats 3d
    for result in ttest_report_3d:
        result_copy = result.copy()
        result_copy['feature_type'] = '3d'
        result_copy['task'] = task_name
        all_ttest_results.append(result_copy)
    
    if all_ttest_results:
        df_ttest = pd.DataFrame(all_ttest_results)
        output_file = Path(res_dir) / f'{task_name}_ttest_results.csv'
        df_ttest.to_csv(output_file, index=False)
        print(f"Résultats t-test sauvegardés dans : {output_file}")
        return output_file
    return None

def save_feature_selection_results_to_csv(feature_details_12h, feature_details_3d, res_dir, task_name):
    """Sauvegarde les scores de sélection des features en CSV."""
    all_feature_scores = []
    
    # Traiter les résultats 12h
    for result in feature_details_12h:
        score_entry = {
            'feature_name': result['feature_name'],
            'avg_f1_score': result['avg_f1_score'],
            'type': result['type'],
            'feature_type': '12h',
            'task': task_name,
            'num_features': len(result['feature_cols'])
        }
        
        # Ajouter les scores détaillés par modèle
        for model_result in result['detailed_results']:
            score_entry[f"{model_result['model']}_accuracy"] = model_result['accuracy']
            score_entry[f"{model_result['model']}_f1"] = model_result['f1']
            score_entry[f"{model_result['model']}_precision"] = model_result['precision']
            score_entry[f"{model_result['model']}_recall"] = model_result['recall']
        
        all_feature_scores.append(score_entry)
    
    # Traiter les résultats 3d
    for result in feature_details_3d:
        score_entry = {
            'feature_name': result['feature_name'],
            'avg_f1_score': result['avg_f1_score'],
            'type': result['type'],
            'feature_type': '3d',
            'task': task_name,
            'num_features': len(result['feature_cols'])
        }
        
        # Ajouter les scores détaillés par modèle
        for model_result in result['detailed_results']:
            score_entry[f"{model_result['model']}_accuracy"] = model_result['accuracy']
            score_entry[f"{model_result['model']}_f1"] = model_result['f1']
            score_entry[f"{model_result['model']}_precision"] = model_result['precision']
            score_entry[f"{model_result['model']}_recall"] = model_result['recall']
        
        all_feature_scores.append(score_entry)
    
    if all_feature_scores:
        df_scores = pd.DataFrame(all_feature_scores)
        output_file = Path(res_dir) / f'{task_name}_feature_selection_scores.csv'
        df_scores.to_csv(output_file, index=False)
        print(f"Scores de sélection des features sauvegardés dans : {output_file}")
        return output_file
    return None

def save_final_model_results_to_csv(results_12h, results_3d, res_dir, task_name):
    """Sauvegarde les scores finaux des modèles avec les meilleures features."""
    all_final_scores = []
    
    # Traiter les résultats 12h
    for result_list in results_12h:
        for model_result in result_list:
            score_entry = model_result.copy()
            score_entry['feature_type'] = '12h'
            score_entry['task'] = task_name
            all_final_scores.append(score_entry)
    
    # Traiter les résultats 3d
    for result_list in results_3d:
        for model_result in result_list:
            score_entry = model_result.copy()
            score_entry['feature_type'] = '3d'
            score_entry['task'] = task_name
            all_final_scores.append(score_entry)
    
    if all_final_scores:
        df_final = pd.DataFrame(all_final_scores)
        output_file = Path(res_dir) / f'{task_name}_final_model_scores.csv'
        df_final.to_csv(output_file, index=False)
        print(f"Scores finaux des modèles sauvegardés dans : {output_file}")
        return output_file
    return None

def build_models():
    """Hyperparamètres d'après l'étude (5 classifieurs)."""
    # Paramètres pour GridSearch AdaBoost
    ada_param_grid = {
        'estimator__max_depth': [2, 3, 4, 5, 6, 7],
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
    
    # AdaBoost avec GridSearchCV
    ada_base = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=4), 
        random_state=RANDOM_STATE
    )
    ada_grid = GridSearchCV(
        ada_base, 
        ada_param_grid, 
        cv=3, 
        scoring='f1_macro',
        n_jobs=-1
    )

    models = {
        "MLP": MLPClassifier(hidden_layer_sizes=(20, 6, 3), activation="relu",
                             max_iter=10000, random_state=RANDOM_STATE),
        "DT": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=7, max_depth=4, random_state=RANDOM_STATE),
        "GBM": GradientBoostingClassifier(n_estimators=1000, learning_rate=0.4,
                                          random_state=RANDOM_STATE),
        "Ada": ada_base,
    }
    return models

def build_regression_models():
    """Hyperparamètres pour les modèles de régression."""
    # Paramètres pour GridSearch AdaBoost Regressor
    ada_reg_param_grid = {
        'estimator__max_depth': [2, 3, 4, 5, 6, 7],
        'n_estimators': [50, 100, 200, 300, 400, 500], 
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
    
    # AdaBoost Regressor avec GridSearchCV
    ada_reg_base = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        random_state=RANDOM_STATE
    )
    ada_reg_grid = GridSearchCV(
        ada_reg_base,
        ada_reg_param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    
    models = {
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=2000, random_state=RANDOM_STATE),
        "DT": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        "Ada": ada_reg_base,
    }
    return models

# Score function: t-test binaire par feature (Welch, robust aux variances inégales)
def ttest_score_func(X, y):
    """
    Retourne (scores, pvalues) pour SelectKBest sur base d'un t-test indépendant.
    - scores: |t| par feature
    - pvalues: p-value correspondante
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("ttest_score_func attend une classification binaire.")
    mask0 = y == classes[0]
    mask1 = y == classes[1]
    t_stat, p_val = ttest_ind(X[mask0], X[mask1], axis=0, equal_var=False, nan_policy="omit")
    scores = np.nan_to_num(np.abs(t_stat), nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(p_val, nan=1.0, posinf=1.0)
    return scores, pvalues

# Rapport t/p par feature pour le reporting (calcul sur l'échantillon équilibré de la tâche)
def compute_ttest_report(X, y, columns, alpha=0.05):
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError("compute_ttest_report attend une classification binaire.")
    mask0 = y_arr == classes[0]
    mask1 = y_arr == classes[1]
    t_stat, p_val = ttest_ind(
        X[columns].values[mask0], X[columns].values[mask1],
        axis=0, equal_var=False, nan_policy="omit"
    )
    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)
    p_val = np.nan_to_num(p_val, nan=1.0, posinf=1.0)
    report = []
    for col, t, p in zip(columns, t_stat, p_val):
        report.append({"feature": col, "t": float(t), "p": float(p), "keep": bool(p < alpha)})
    return report

def balance_by_undersampling(X, y):
    """Sous-échantillonne la classe majoritaire pour équilibrer (simple, déterministe)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Cette fonction attend une tâche binaire.")
    minority = classes[np.argmin(counts)]
    majority = classes[np.argmax(counts)]
    n = counts.min()

    idx_min = np.where(y == minority)[0]
    idx_maj = np.where(y == majority)[0]

    rng = np.random.RandomState(RANDOM_STATE)
    keep_maj = rng.choice(idx_maj, size=n, replace=False)
    keep = np.concatenate([idx_min, keep_maj])
    keep.sort()
    return X.iloc[keep].reset_index(drop=True), pd.Series(y[keep])

def evaluate_feature_combination(X, y, feature_subset, models, cv):
    """Évalue une combinaison de features donnée avec les modèles et retourne les scores moyens."""
    # Filtrer les colonnes pour ne conserver que celles présentes dans X
    valid_features = [col for col in feature_subset if col in X.columns]
    missing_features = [col for col in feature_subset if col not in X.columns]
    if missing_features:
        print(f"Avertissement : les colonnes suivantes sont absentes et seront ignorées : {missing_features}")

    if not valid_features:
        raise ValueError("Aucune des colonnes dans feature_subset n'est présente dans les données.")

    fold_results = []
    for model_name, model in models.items():
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx][valid_features], X.iloc[test_idx][valid_features]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline([("clf", model)])
            pipe.fit(X_tr, y_tr)

            y_hat = pipe.predict(X_te)
            fold_metrics.append({
                "accuracy": accuracy_score(y_te, y_hat),
                "precision": precision_score(y_te, y_hat, average="macro", zero_division=0),
                "recall": recall_score(y_te, y_hat, average="macro", zero_division=0),
                "f1": f1_score(y_te, y_hat, average="macro", zero_division=0),
            })
        avg_metrics = {m: float(np.mean([fm[m] for fm in fold_metrics])) for m in fold_metrics[0]}
        fold_results.append({"model": model_name, **avg_metrics})
    return fold_results

def test_combinations_in_parallel(X, y, feature_group, models, cv):
    """
    Teste toutes les combinaisons de features dans un groupe donné en parallèle.
    """
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_feature_combination)(X, y, feature_subset, models, cv)
        for r in range(1, len(feature_group) + 1)
        for feature_subset in combinations(feature_group, r)
    )
    return results

def evaluate_single_feature(X, y, feature_cols, models,folds=2):
    """
    Évalue une feature ou un groupe de features avec folding et retourne les scores moyens.
    """
    if folds == 'loo':
        loo = LeaveOneOut()
    #else, if folds is a integer, use StratifiedKFold with that number of splits
    elif isinstance(folds, int) and folds > 1:
        loo = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        raise ValueError("folds doit être 'loo' ou un entier > 1.")
        
    all_results = []
    
    for model_name, model in models.items():
        fold_metrics = []
        fold_iterator = loo.split(X) if folds == 'loo' else loo.split(X, y)
        for train_idx, test_idx in fold_iterator:
            X_tr, X_te = X.iloc[train_idx][feature_cols], X.iloc[test_idx][feature_cols]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            try:
                pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", model)])
                pipe.fit(X_tr, y_tr)
                y_hat = pipe.predict(X_te)
                
                fold_metrics.append({
                    "accuracy": accuracy_score(y_te, y_hat),
                    "precision": precision_score(y_te, y_hat, average="macro", zero_division=0),
                    "recall": recall_score(y_te, y_hat, average="macro", zero_division=0),
                    "f1": f1_score(y_te, y_hat, average="macro", zero_division=0),
                })
            except Exception as e:
                # Si le modèle échoue, attribuer des scores de 0
                fold_metrics.append({
                    "accuracy": 0.0,
                    "precision": 0.0, 
                    "recall": 0.0,
                    "f1": 0.0,
                })
        
        avg_metrics = {m: float(np.mean([fm[m] for fm in fold_metrics])) for m in fold_metrics[0]}
        all_results.append({"model": model_name, **avg_metrics})
    
    # Retourner la moyenne des f1-scores de tous les modèles comme score principal
    avg_f1 = np.mean([r["f1"] for r in all_results])
    return avg_f1, all_results

def group_12h_features(feature_cols):
    """
    Regroupe les features 12h par nom de base (sans le suffixe numérique).
    Ex: acti_activity_mean_12h_0, acti_activity_mean_12h_1, ... -> acti_activity_mean_12h
    """
    feature_groups = {}
    other_features = []
    
    for col in feature_cols:
        if '12h_' in col:
            # Extraire le nom de base sans le suffixe numérique
            base_name = '_'.join(col.split('_')[:-1])  # Enlever le dernier élément (le numéro)
            if base_name not in feature_groups:
                feature_groups[base_name] = []
            feature_groups[base_name].append(col)
        else:
            other_features.append(col)
    
    return feature_groups, other_features

def select_best_features_loo(X, y, feature_cols, models, top_k=10):
    """
    Sélectionne les meilleures features basées sur leur performance individuelle en Leave-One-Out.
    Pour les features 12h, les teste par groupe de 6 valeurs.
    """
    print("=== SÉLECTION DES FEATURES PAR LEAVE-ONE-OUT ===")
    
    # Regrouper les features 12h
    feature_groups_12h, other_features = group_12h_features(feature_cols)
    
    feature_results = []
    
    # Tester les groupes de features 12h
    print(f"Test des {len(feature_groups_12h)} groupes de features 12h...")
    for base_name, group_features in feature_groups_12h.items():
        print(f"Test du groupe {base_name} ({len(group_features)} features)")
        avg_f1, detailed_results = evaluate_single_feature(X, y, group_features, models)
        feature_results.append({
            "feature_name": base_name,
            "feature_cols": group_features,
            "avg_f1_score": avg_f1,
            "detailed_results": detailed_results,
            "type": "12h_group"
        })
    
    # Tester les autres features individuellement
    print(f"Test des {len(other_features)} autres features...")
    for feature in other_features:
        print(f"Test de {feature}")
        avg_f1, detailed_results = evaluate_single_feature(X, y, [feature], models)
        feature_results.append({
            "feature_name": feature,
            "feature_cols": [feature],
            "avg_f1_score": avg_f1,
            "detailed_results": detailed_results,
            "type": "individual"
        })
    
    # Trier par performance décroissante
    feature_results.sort(key=lambda x: x["avg_f1_score"], reverse=True)
    
    # Sélectionner les top_k meilleures
    selected_features = feature_results[:top_k]
    selected_feature_cols = []
    for feat in selected_features:
        selected_feature_cols.extend(feat["feature_cols"])
    
    print(f"\n=== TOP {top_k} FEATURES SÉLECTIONNÉES ===")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. {feat['feature_name']} (F1={feat['avg_f1_score']:.4f}, type={feat['type']})")
    
    return selected_feature_cols, feature_results

def select_features_by_correlation_significance(correlation_results, min_features=3):
    """Sélectionne les features basées sur la significativité des corrélations."""
    
    # Séparer les features par type
    features_12h = [r for r in correlation_results if '12h' in r['feature']]
    features_3d = [r for r in correlation_results if r['feature'].endswith('3d')]
    
    selected_features = {'12h': [], '3d': []}
    
    for feature_type, features in [('12h', features_12h), ('3d', features_3d)]:
        # Trier par p-value croissante (plus significatif d'abord)
        features_sorted = sorted(features, key=lambda x: x['pearson_pval'])
        
        # Sélectionner les features significatives
        significant_features = [f for f in features_sorted if f['significant_pearson']]
        
        if len(significant_features) >= min_features:
            selected_features[feature_type] = [f['feature'] for f in significant_features]
            print(f"Features {feature_type} sélectionnées (significatives): {len(significant_features)}")
        else:
            # Si moins de min_features significatives, prendre les min_features meilleures
            top_features = features_sorted[:min_features]
            selected_features[feature_type] = [f['feature'] for f in top_features]
            print(f"Features {feature_type} sélectionnées (top {min_features}): {len(top_features)}")
    
    return selected_features

def evaluate_regression_models(X, y, feature_subset, models, cv):
    """Évalue les modèles de régression avec les features données."""
    
    # Filtrer les colonnes pour ne conserver que celles présentes dans X
    valid_features = [col for col in feature_subset if col in X.columns]
    
    if not valid_features:
        return []
    
    fold_results = []
    for model_name, model in models.items():
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx][valid_features], X.iloc[test_idx][valid_features]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            try:
                # Pipeline avec normalisation pour la régression
                pipe = Pipeline([("scaler", MinMaxScaler()), ("reg", model)])
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                
                # Métriques de régression
                mae = mean_absolute_error(y_te, y_pred)
                mse = mean_squared_error(y_te, y_pred)
                rmse = np.sqrt(mse)
                
                # R² avec gestion des cas particuliers
                try:
                    r2 = r2_score(y_te, y_pred)
                except:
                    r2 = 0.0
                
                fold_metrics.append({
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                })
            except Exception as e:
                # Si le modèle échoue, attribuer des scores par défaut
                fold_metrics.append({
                    "mae": np.inf,
                    "mse": np.inf, 
                    "rmse": np.inf,
                    "r2": -np.inf
                })
        
        # Calculer les moyennes des métriques
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [fm[metric] for fm in fold_metrics if not np.isinf(fm[metric]) and not np.isneginf(fm[metric])]
            if values:
                avg_metrics[metric] = float(np.mean(values))
            else:
                avg_metrics[metric] = np.inf if metric != 'r2' else -np.inf
        
        fold_results.append({"model": model_name, **avg_metrics})
    
    return fold_results

def generate_markdown_report(ttest_report_12h, ttest_report_3d, feature_details_12h, feature_details_3d, 
                           results_12h, results_3d, selected_12h, selected_3d, res_dir, task_name,
                           combined_results=None, regression_results=None):
    """Génère un rapport markdown complet."""
    
    report_lines = [
        f"# Rapport d'analyse - {task_name}",
        "",
        f"Date de génération : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Résumé des analyses",
        "",
    ]
    
    # Features significatives au t-test
    significant_12h = [r for r in ttest_report_12h if r['keep']]
    significant_3d = [r for r in ttest_report_3d if r['keep']]
    
    report_lines.extend([
        "### Features significatives (p < 0.05)",
        "",
        f"**Features 12h significatives :** {len(significant_12h)}/{len(ttest_report_12h)}",
        ""
    ])
    
    if significant_12h:
        report_lines.append("| Feature | t-statistic | p-value |")
        report_lines.append("|---------|-------------|---------|")
        for feat in sorted(significant_12h, key=lambda x: x['p'])[:10]:  # Top 10
            report_lines.append(f"| {feat['feature']} | {feat['t']:.4f} | {feat['p']:.6f} |")
        report_lines.append("")
    
    report_lines.extend([
        f"**Features 3d significatives :** {len(significant_3d)}/{len(ttest_report_3d)}",
        ""
    ])
    
    if significant_3d:
        report_lines.append("| Feature | t-statistic | p-value |")
        report_lines.append("|---------|-------------|---------|")
        for feat in sorted(significant_3d, key=lambda x: x['p'])[:10]:  # Top 10
            report_lines.append(f"| {feat['feature']} | {feat['t']:.4f} | {feat['p']:.6f} |")
        report_lines.append("")
    
    # Top 10 groupes de features
    all_features = feature_details_12h + feature_details_3d
    top_features = sorted(all_features, key=lambda x: x['avg_f1_score'], reverse=True)[:10]
    
    report_lines.extend([
        "### Top 10 groupes/features par performance F1",
        "",
        "| Rang | Feature/Groupe | Type | F1 Score | Détails |",
        "|------|----------------|------|----------|---------|"
    ])
    
    for i, feat in enumerate(top_features, 1):
        feature_type = feat.get('type', 'unknown')
        num_features = len(feat['feature_cols'])
        details = f"{num_features} feature(s)" if feature_type == '12h_group' else "feature individuelle"
        report_lines.append(
            f"| {i} | {feat['feature_name']} | {feature_type} | {feat['avg_f1_score']:.4f} | {details} |"
        )
    
    report_lines.append("")
    
    # Scores finaux avec meilleures features
    report_lines.extend([
        "### Performance des modèles avec les meilleures features",
        "",
        "#### Features 12h sélectionnées",
        f"**Nombre de features :** {len(selected_12h)}",
        ""
    ])
    
    if results_12h and len(results_12h) > 0:
        report_lines.extend([
            "| Modèle | Accuracy | Precision | Recall | F1-Score |",
            "|--------|----------|-----------|--------|----------|"
        ])
        for model_result in results_12h[0]:
            report_lines.append(
                f"| {model_result['model']} | {model_result['accuracy']:.4f} | "
                f"{model_result['precision']:.4f} | {model_result['recall']:.4f} | "
                f"{model_result['f1']:.4f} |"
            )
        report_lines.append("")
    
    report_lines.extend([
        "#### Features 3d sélectionnées",
        f"**Nombre de features :** {len(selected_3d)}",
        ""
    ])
    
    if results_3d and len(results_3d) > 0:
        report_lines.extend([
            "| Modèle | Accuracy | Precision | Recall | F1-Score |",
            "|--------|----------|-----------|--------|----------|"
        ])
        for model_result in results_3d[0]:
            report_lines.append(
                f"| {model_result['model']} | {model_result['accuracy']:.4f} | "
                f"{model_result['precision']:.4f} | {model_result['recall']:.4f} | "
                f"{model_result['f1']:.4f} |"
            )
        report_lines.append("")
    
    # Ajout des résultats combinés
    if combined_results:
        report_lines.extend([
            "#### Features combinées (12h + 3d)",
            f"**Nombre total de features :** {len(selected_12h) + len(selected_3d)}",
            ""
        ])
        
        report_lines.extend([
            "| Modèle | Accuracy | Precision | Recall | F1-Score |",
            "|--------|----------|-----------|--------|----------|"
        ])
        for model_result in combined_results:
            report_lines.append(
                f"| {model_result['model']} | {model_result['accuracy']:.4f} | "
                f"{model_result['precision']:.4f} | {model_result['recall']:.4f} | "
                f"{model_result['f1']:.4f} |"
            )
        report_lines.append("")
    
    # Meilleurs scores globaux
    all_final_results = []
    if results_12h:
        all_final_results.extend(results_12h[0])
    if results_3d:
        all_final_results.extend(results_3d[0])
    if combined_results:
        all_final_results.extend(combined_results)
    
    if all_final_results:
        best_f1 = max(all_final_results, key=lambda x: x['f1'])
        best_accuracy = max(all_final_results, key=lambda x: x['accuracy'])
        
        report_lines.extend([
            "### Meilleurs résultats globaux",
            "",
            f"**Meilleur F1-Score :** {best_f1['f1']:.4f} ({best_f1['model']})",
            f"**Meilleure Accuracy :** {best_accuracy['accuracy']:.4f} ({best_accuracy['model']})",
            "",
        ])
    
    # Ajout des résultats de régression/corrélation
    if regression_results:
        report_lines.extend([
            "### Analyse de régression avec modèles",
            "",
        ])
        
        for target, reg_data in regression_results.items():
            correlations = reg_data.get('correlations', [])
            selected_features = reg_data.get('selected_features', {})
            model_results = reg_data.get('model_results', {})
            
            significant_corr = [r for r in correlations if r['significant_pearson']]
            
            report_lines.extend([
                f"#### {target.upper()} - Analyse complète",
                f"**Corrélations significatives :** {len(significant_corr)}/{len(correlations)}",
                f"**Features sélectionnées 12h :** {len(selected_features.get('12h', []))}",
                f"**Features sélectionnées 3d : ** {len(selected_features.get('3d', []))}",
                ""
            ])
            
            # Tableau des meilleures corrélations
            if significant_corr:
                report_lines.extend([
                    "**Top corrélations significatives :**",
                    "",
                    "| Feature | Corrélation (Pearson) | p-value |",
                    "|---------|----------------------|---------|"
                ])
                for feat in sorted(significant_corr, key=lambda x: abs(x['pearson_corr']), reverse=True)[:5]:
                    report_lines.append(f"| {feat['feature']} | {feat['pearson_corr']:.4f} | {feat['pearson_pval']:.6f} |")
                report_lines.append("")
            
            # Résultats des modèles de régression
            if model_results:
                report_lines.extend([
                    "**Performance des modèles de régression :**",
                    "",
                    "| Type Features | Meilleur Modèle | R² | RMSE | MAE |",
                    "|---------------|-----------------|----|----- |----|"
                ])
                
                for feature_type in ['12h', '3d', 'combined']:
                    if feature_type in model_results and model_results[feature_type]:
                        results = model_results[feature_type]
                        best_model = max(results, key=lambda x: x['r2'] if not np.isinf(x['r2']) else -np.inf)
                        report_lines.append(
                            f"| {feature_type} | {best_model['model']} | "
                            f"{best_model['r2']:.4f} | {best_model['rmse']:.4f} | {best_model['mae']:.4f} |"
                        )
                
                report_lines.append("")
    
    # Sauvegarder le rapport
    report_content = "\n".join(report_lines)
    output_file = Path(res_dir) / f'{task_name}_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Rapport markdown sauvegardé dans : {output_file}")
    return output_file

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

def save_significant_features_to_json(ttest_report_12h, ttest_report_3d, feature_details_12h, feature_details_3d,
                                     selected_12h, selected_3d, correlation_results, res_dir, task_name):
    """Sauvegarde les features significatives dans un fichier JSON."""
    
    # Features significatives au t-test
    significant_12h_ttest = [r['feature'] for r in ttest_report_12h if r['keep']]
    significant_3d_ttest = [r['feature'] for r in ttest_report_3d if r['keep']]
    
    # Top features par performance F1
    top_features_12h = sorted(feature_details_12h, key=lambda x: x['avg_f1_score'], reverse=True)[:10]
    top_features_3d = sorted(feature_details_3d, key=lambda x: x['avg_f1_score'], reverse=True)[:10]
    
    significant_data = {
        'task': task_name,
        'generation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'classification': {
            'ttest_significant': {
                '12h': {
                    'count': len(significant_12h_ttest),
                    'features': significant_12h_ttest
                },
                '3d': {
                    'count': len(significant_3d_ttest),
                    'features': significant_3d_ttest
                }
            },
            'selected_for_modeling': {
                '12h': {
                    'count': len(selected_12h),
                    'features': selected_12h
                },
                '3d': {
                    'count': len(selected_3d),
                    'features': selected_3d
                },
                'combined': {
                    'count': len(selected_12h) + len(selected_3d),
                    'features': selected_12h + selected_3d
                }
            },
            'top_performers': {
                '12h': [
                    {
                        'feature_name': f['feature_name'],
                        'f1_score': f['avg_f1_score'],
                        'type': f['type'],
                        'num_features': len(f['feature_cols'])
                    }
                    for f in top_features_12h
                ],
                '3d': [
                    {
                        'feature_name': f['feature_name'],
                        'f1_score': f['avg_f1_score'],
                        'type': f['type'],
                        'num_features': len(f['feature_cols'])
                    }
                    for f in top_features_3d
                ]
            }
        }
    }
    
    # Ajouter les résultats de corrélation si disponibles
    if correlation_results:
        significant_data['regression'] = {}
        for target, reg_data in correlation_results.items():
            correlations = reg_data.get('correlations', [])
            selected_features = reg_data.get('selected_features', {})
            
            significant_corr = [r for r in correlations if r['significant_pearson']]
            
            significant_data['regression'][target] = {
                'significant_correlations': {
                    'count': len(significant_corr),
                    'features': [
                        {
                            'feature': r['feature'],
                            'pearson_corr': r['pearson_corr'],
                            'pearson_pval': r['pearson_pval'],
                            'spearman_corr': r['spearman_corr'],
                            'spearman_pval': r['spearman_pval']
                        }
                        for r in sorted(significant_corr, key=lambda x: abs(x['pearson_corr']), reverse=True)
                    ]
                },
                'selected_for_modeling': {
                    '12h': {
                        'count': len(selected_features.get('12h', [])),
                        'features': selected_features.get('12h', [])
                    },
                    '3d': {
                        'count': len(selected_features.get('3d', [])),
                        'features': selected_features.get('3d', [])
                    }
                },
                'top_correlations': [
                    {
                        'feature': r['feature'],
                        'pearson_corr': r['pearson_corr'],
                        'pearson_pval': r['pearson_pval']
                    }
                    for r in correlations[:20]
                ]
            }
    
    # Sauvegarder en JSON
    output_file = Path(res_dir) / f'{task_name}_significant_features.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(significant_data, f, indent=2, ensure_ascii=False)
    
    print(f"Features significatives sauvegardées dans : {output_file}")
    return output_file

def run_task(df, feature_cols, target_col, res_dir='/home/ndecaux/Code/actiDep/analysis'):
    """Exécute la tâche binaire demandée avec sélection des meilleures features par LOO."""
    # Créer le dossier de résultats s'il n'existe pas
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    data = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    X = data[feature_cols]
    y = data[target_col]

    # Sous-échantillonnage pour équilibrer (comme dans l'étude)
    X_bal, y_bal = X, y  # balance_by_undersampling(X, y)

    # Résidualisation des features
    X_bal_residualized = X_bal.copy()
    for f in feature_cols:
        if target_col == 'group':
            X_bal_residualized[f] = ols_residualize(X_bal[f], df[["age","sex","city"]])
        else:
            X_bal_residualized[f] = ols_residualize(X_bal[f], df[["age","sex","city",'duration_dep','type_dep']])

    # Séparer les features en deux groupes : 12h et 3d
    features_12h = [col for col in feature_cols if '12h' in col]
    features_3d = [col for col in feature_cols if col.endswith("3d")]

    models = build_models()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    results_12h, feature_details_12h, selected_12h = smarter_feature_selection(X_bal_residualized, y_bal, features_12h, models, cv)
    results_3d, feature_details_3d, selected_3d = smarter_feature_selection(X_bal_residualized, y_bal, features_3d, models, cv)

    # Évaluation des features combinées
    combined_results = evaluate_combined_features(X_bal_residualized, y_bal, selected_12h, selected_3d, models, cv)

    # Rapport t-test pour les features sélectionnées
    ttest_report_12h = compute_ttest_report(X_bal_residualized, y_bal, selected_12h) if selected_12h else []
    ttest_report_3d = compute_ttest_report(X_bal_residualized, y_bal, selected_3d) if selected_3d else []
    
    # Sauvegarder les résultats en fichiers
    task_name = f"{target_col}_analysis"
    
    save_ttest_results_to_csv(ttest_report_12h, ttest_report_3d, res_dir, task_name)
    save_feature_selection_results_to_csv(feature_details_12h, feature_details_3d, res_dir, task_name)
    save_final_model_results_to_csv(results_12h, results_3d, res_dir, task_name)
    
    if combined_results:
        save_combined_results_to_csv(combined_results, res_dir, task_name)
    
    # Sauvegarder les features significatives en JSON
    save_significant_features_to_json(ttest_report_12h, ttest_report_3d, feature_details_12h, feature_details_3d,
                                     selected_12h, selected_3d, None, res_dir, task_name)
    
    generate_markdown_report(ttest_report_12h, ttest_report_3d, feature_details_12h, feature_details_3d,
                           results_12h, results_3d, selected_12h, selected_3d, res_dir, task_name,
                           combined_results=combined_results)
    
    report = {
        "ttest_report_12h": ttest_report_12h,
        "ttest_report_3d": ttest_report_3d,
        "results_12h": results_12h,
        "results_3d": results_3d,
        "feature_selection_details_12h": feature_details_12h,
        "feature_selection_details_3d": feature_details_3d,
        "selected_features_12h": selected_12h,
        "selected_features_3d": selected_3d,
        "combined_results": combined_results,
    }
    return report

def smarter_feature_selection(X, y, feature_cols, models, cv):
    """
    Sélectionne les features en utilisant Leave-One-Out pour évaluer chaque feature individuellement.
    """
    # Sélectionner les meilleures features avec LOO
    selected_features, all_feature_results = select_best_features_loo(X, y, feature_cols, models, top_k=10)
    
    if not selected_features:
        raise ValueError("Aucune feature sélectionnée après le processus de sélection.")

    # Évaluer les features sélectionnées ensemble
    result = evaluate_feature_combination(X, y, selected_features, models, cv)
    
    return [result], all_feature_results, selected_features

def evaluate_combined_features(X, y, selected_12h, selected_3d, models, cv):
    """Évalue la performance des meilleures features 12h et 3d combinées."""
    combined_features = selected_12h + selected_3d
    
    if not combined_features:
        return None
        
    print(f"=== ÉVALUATION DES FEATURES COMBINÉES ===")
    print(f"Features 12h sélectionnées: {len(selected_12h)}")
    print(f"Features 3d sélectionnées: {len(selected_3d)}")
    print(f"Total features combinées: {len(combined_features)}")
    
    # Évaluer les features combinées
    combined_results = evaluate_feature_combination(X, y, combined_features, models, cv)
    
    return combined_results

def generate_regression_markdown_report(correlation_results, selected_features, model_results, res_dir, target_col):
    """Génère un rapport markdown spécifique pour l'analyse de régression."""
    
    report_lines = [
        f"# Rapport d'analyse de régression - {target_col.upper()}",
        "",
        f"Date de génération : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Résumé de l'analyse",
        "",
        f"**Target :** {target_col}",
        f"**Nombre total de features analysées :** {len(correlation_results)}",
        "",
    ]
    
    # Corrélations significatives
    significant_corr = [r for r in correlation_results if r['significant_pearson']]
    report_lines.extend([
        "### Corrélations significatives (p < 0.05)",
        "",
        f"**Nombre de corrélations significatives :** {len(significant_corr)}/{len(correlation_results)}",
        ""
    ])
    
    if significant_corr:
        report_lines.extend([
            "| Rang | Feature | Corrélation (Pearson) | p-value | Corrélation (Spearman) | p-value Spearman |",
            "|------|---------|----------------------|---------|------------------------|------------------|"
        ])
        for i, feat in enumerate(sorted(significant_corr, key=lambda x: abs(x['pearson_corr']), reverse=True), 1):
            report_lines.append(
                f"| {i} | {feat['feature']} | {feat['pearson_corr']:.4f} | {feat['pearson_pval']:.6f} | "
                f"{feat['spearman_corr']:.4f} | {feat['spearman_pval']:.6f} |"
            )
        report_lines.append("")
    
    # Top corrélations (incluant non-significatives)
    report_lines.extend([
        "### Top 20 corrélations (toutes)",
        "",
        "| Rang | Feature | Corrélation (Pearson) | p-value | Significative |",
        "|------|---------|----------------------|---------|---------------|"
    ])
    
    for i, feat in enumerate(correlation_results[:20], 1):
        significant_mark = "✓" if feat['significant_pearson'] else "✗"
        report_lines.append(
            f"| {i} | {feat['feature']} | {feat['pearson_corr']:.4f} | "
            f"{feat['pearson_pval']:.6f} | {significant_mark} |"
        )
    report_lines.append("")
    
    # Features sélectionnées par type
    report_lines.extend([
        "### Features sélectionnées pour la modélisation",
        "",
        f"**Features 12h sélectionnées :** {len(selected_features.get('12h', []))}",
        f"**Features 3d sélectionnées :** {len(selected_features.get('3d', []))}",
        ""
    ])
    
    for feature_type in ['12h', '3d']:
        features = selected_features.get(feature_type, [])
        if features:
            report_lines.extend([
                f"#### Features {feature_type} sélectionnées",
                ""
            ])
            for i, feature in enumerate(features, 1):
                # Trouver les détails de corrélation pour cette feature
                feature_details = next((r for r in correlation_results if r['feature'] == feature), None)
                if feature_details:
                    report_lines.append(
                        f"{i}. **{feature}** - r={feature_details['pearson_corr']:.4f}, "
                        f"p={feature_details['pearson_pval']:.6f}"
                    )
                else:
                    report_lines.append(f"{i}. **{feature}**")
            report_lines.append("")
    
    # Résultats des modèles
    if model_results:
        report_lines.extend([
            "### Performance des modèles de régression",
            "",
            "| Type Features | Modèle | R² | RMSE | MAE |",
            "|---------------|--------|----|----- |----|"
        ])
        
        for feature_type in ['12h', '3d', 'combined']:
            if feature_type in model_results and model_results[feature_type]:
                results = model_results[feature_type]
                for result in sorted(results, key=lambda x: x['r2'] if not np.isinf(x['r2']) else -np.inf, reverse=True):
                    report_lines.append(
                        f"| {feature_type} | {result['model']} | "
                        f"{result['r2']:.4f} | {result['rmse']:.4f} | {result['mae']:.4f} |"
                    )
        report_lines.append("")
        
        # Meilleur modèle global
        all_results = []
        for feature_type, results in model_results.items():
            if results:
                all_results.extend(results)
        
        if all_results:
            best_model = max(all_results, key=lambda x: x['r2'] if not np.isinf(x['r2']) else -np.inf)
            report_lines.extend([
                "### Meilleur modèle global",
                "",
                f"**Modèle :** {best_model['model']}",
                f"**Type de features :** {next(ft for ft, res in model_results.items() if best_model in res)}",
                f"**R² :** {best_model['r2']:.4f}",
                f"**RMSE :** {best_model['rmse']:.4f}",
                f"**MAE :** {best_model['mae']:.4f}",
                ""
            ])
    
    # Statistiques de sélection
    report_lines.extend([
        "### Statistiques de sélection des features",
        "",
        f"**Critère de sélection :** Corrélations significatives (p < 0.05) ou top 3 par type",
        f"**Features 12h significatives :** {len([f for f in correlation_results if '12h' in f['feature'] and f['significant_pearson']])}",
        f"**Features 3d significatives :** {len([f for f in correlation_results if f['feature'].endswith('3d') and f['significant_pearson']])}",
        ""
    ])
    
    # Distribution des p-values
    p_values = [r['pearson_pval'] for r in correlation_results]
    p_ranges = [
        ("< 0.001", len([p for p in p_values if p < 0.001])),
        ("0.001-0.01", len([p for p in p_values if 0.001 <= p < 0.01])),
        ("0.01-0.05", len([p for p in p_values if 0.01 <= p < 0.05])),
        ("0.05-0.1", len([p for p in p_values if 0.05 <= p < 0.1])),
        (">= 0.1", len([p for p in p_values if p >= 0.1]))
    ]
    
    report_lines.extend([
        "### Distribution des p-values",
        "",
        "| Plage p-value | Nombre de features |",
        "|---------------|-------------------|"
    ])
    
    for range_name, count in p_ranges:
        report_lines.append(f"| {range_name} | {count} |")
    
    report_lines.append("")
    
    # Sauvegarder le rapport
    report_content = "\n".join(report_lines)
    output_file = Path(res_dir) / f'{target_col}_regression_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Rapport de régression markdown sauvegardé dans : {output_file}")
    return output_file

def analyze_regression_correlation(df, feature_cols, target_col, res_dir):
    """Analyse les corrélations pour les tâches de régression (ami, aes) avec évaluation des modèles."""
    print(f"\n=== ANALYSE DE RÉGRESSION POUR {target_col.upper()} ===")
    
    data = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    X = data[feature_cols]
    y = data[target_col]
    
    correlation_results = []
    
    # Calculer les corrélations pour chaque feature
    for feature in feature_cols:
        if feature in X.columns:
            # Corrélation de Pearson
            try:
                pearson_corr, pearson_pval = pearsonr(X[feature], y)
            except:
                pearson_corr, pearson_pval = np.nan, np.nan
            
            # Corrélation de Spearman  
            try:
                spearman_corr, spearman_pval = spearmanr(X[feature], y)
            except:
                spearman_corr, spearman_pval = np.nan, np.nan
            
            correlation_results.append({
                'feature': feature,
                'pearson_corr': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                'pearson_pval': float(pearson_pval) if not np.isnan(pearson_pval) else 1.0,
                'spearman_corr': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                'spearman_pval': float(spearman_pval) if not np.isnan(spearman_pval) else 1.0,
                'significant_pearson': bool(pearson_pval < 0.05) if not np.isnan(pearson_pval) else False,
                'significant_spearman': bool(spearman_pval < 0.05) if not np.isnan(spearman_pval) else False,
                'target': target_col
            })
    
    # Trier par corrélation absolue décroissante (Pearson)
    correlation_results.sort(key=lambda x: abs(x['pearson_corr']), reverse=True)
    
    # Sélectionner les meilleures features par groupe
    selected_features = select_features_by_correlation_significance(correlation_results, min_features=3)
    
    # Évaluer les modèles de régression
    regression_models = build_regression_models()
    # Utiliser KFold au lieu de StratifiedKFold pour la régression
    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=RANDOM_STATE)
    
    model_results = {}
    
    # Évaluer les features 12h
    if selected_features['12h']:
        print(f"Évaluation des modèles avec features 12h ({len(selected_features['12h'])} features)")
        results_12h = evaluate_regression_models(X, y, selected_features['12h'], regression_models, cv)
        model_results['12h'] = results_12h
    
    # Évaluer les features 3d
    if selected_features['3d']:
        print(f"Évaluation des modèles avec features 3d ({len(selected_features['3d'])} features)")
        results_3d = evaluate_regression_models(X, y, selected_features['3d'], regression_models, cv)
        model_results['3d'] = results_3d
    
    # Évaluer les features combinées
    combined_features = selected_features['12h'] + selected_features['3d']
    if combined_features:
        print(f"Évaluation des modèles avec features combinées ({len(combined_features)} features)")
        results_combined = evaluate_regression_models(X, y, combined_features, regression_models, cv)
        model_results['combined'] = results_combined
    
    # Sauvegarder les corrélations en CSV
    if correlation_results:
        df_corr = pd.DataFrame(correlation_results)
        output_file = Path(res_dir) / f'{target_col}_correlation_analysis.csv'
        df_corr.to_csv(output_file, index=False)
        print(f"Analyse de corrélation sauvegardée dans : {output_file}")
    
    # Sauvegarder les résultats des modèles
    if model_results:
        all_model_results = []
        for feature_type, results in model_results.items():
            for result in results:
                result_copy = result.copy()
                result_copy['feature_type'] = feature_type
                result_copy['target'] = target_col
                all_model_results.append(result_copy)
        
        if all_model_results:
            df_models = pd.DataFrame(all_model_results)
            output_file = Path(res_dir) / f'{target_col}_regression_model_results.csv'
            df_models.to_csv(output_file, index=False)
            print(f"Résultats des modèles de régression sauvegardés dans : {output_file}")
    
    # Générer le rapport markdown spécifique
    generate_regression_markdown_report(correlation_results, selected_features, model_results, res_dir, target_col)
    
    # Afficher le top 10 des corrélations
    print(f"Top 10 corrélations pour {target_col}:")
    for i, result in enumerate(correlation_results[:10], 1):
        print(f"{i}. {result['feature']}: r={result['pearson_corr']:.4f} (p={result['pearson_pval']:.6f})")
    
    # Afficher les meilleurs modèles
    if model_results:
        print(f"\nMeilleurs modèles par type de features:")
        for feature_type, results in model_results.items():
            if results:
                # Trier par R² décroissant
                best_model = max(results, key=lambda x: x['r2'] if not np.isinf(x['r2']) else -np.inf)
                print(f"  {feature_type}: {best_model['model']} (R²={best_model['r2']:.4f}, RMSE={best_model['rmse']:.4f})")
    
    return {
        'correlations': correlation_results,
        'selected_features': selected_features,
        'model_results': model_results
    }

def save_combined_results_to_csv(combined_results, res_dir, task_name):
    """Sauvegarde les résultats des features combinées."""
    if not combined_results:
        return None
        
    combined_scores = []
    for model_result in combined_results:
        score_entry = model_result.copy()
        score_entry['feature_type'] = 'combined'
        score_entry['task'] = task_name
        combined_scores.append(score_entry)
    
    if combined_scores:
        df_combined = pd.DataFrame(combined_scores)
        output_file = Path(res_dir) / f'{task_name}_combined_features_scores.csv'
        df_combined.to_csv(output_file, index=False)
        print(f"Scores des features combinées sauvegardés dans : {output_file}")
        return output_file
    return None

def main_glm(df):
    # Toutes les features "acti_*" construites en amont
    feature_cols = [c for c in df.columns if c.startswith("acti_")]
    results = {}

    # Tâche 1 : AMI
    results["AMI"] = run_task_glm(df, feature_cols, target_col="ami")

    # Tâche 2 : AES
    results["AES"] = run_task_glm(df, feature_cols, target_col="aes")

    return results
def main(df):
    # Toutes les features "acti_*" construites en amont
    feature_cols = [c for c in df.columns if c.startswith("acti_")]
    results = {}  # S'assurer que c'est un dictionnaire
    
    # Créer le dossier d'analyse s'il n'existe pas
    res_dir = "/home/ndecaux/Code/actiDep/analysis"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    # Tâche 1 : Depressed vs Control (group)
    results["depressed_vs_control"] = run_task(df, feature_cols, target_col="group", res_dir=res_dir)

    # Tâche 2 : Apathy vs Non-apathy (apathy)  
    results["apathy_vs_nonapathy"] = run_task(df, feature_cols, target_col="apathy", res_dir=res_dir)

    # Tâches de régression/corrélation avec modèles
    regression_results = {}
    
    # Analyse AMI
    if 'ami' in df.columns:
        print("Analyse AMI en cours...")
        ami_analysis = analyze_regression_correlation(df, feature_cols, 'ami', res_dir)
        regression_results['ami'] = ami_analysis
        
        # Sauvegarder les features significatives pour AMI
        save_significant_features_to_json([], [], [], [], [], [], 
                                         {'ami': ami_analysis}, res_dir, 'ami_regression')
    
    # Analyse AES  
    if 'aes' in df.columns:
        print("Analyse AES en cours...")
        aes_analysis = analyze_regression_correlation(df, feature_cols, 'aes', res_dir)
        regression_results['aes'] = aes_analysis
        
        # Sauvegarder les features significatives pour AES
        save_significant_features_to_json([], [], [], [], [], [], 
                                         {'aes': aes_analysis}, res_dir, 'aes_regression')
    
    # Générer un JSON global avec toutes les features significatives
    if regression_results or 'depressed_vs_control' in results or 'apathy_vs_nonapathy' in results:
        global_significant = {
            'generation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tasks': {}
        }
        
        # Ajouter les résultats de classification
        for task_name in ['depressed_vs_control', 'apathy_vs_nonapathy']:
            if task_name in results:
                task_data = results[task_name]
                global_significant['tasks'][task_name] = {
                    'type': 'classification',
                    'selected_features_12h': task_data.get('selected_features_12h', []),
                    'selected_features_3d': task_data.get('selected_features_3d', []),
                    'n_significant_12h': len([r for r in task_data.get('ttest_report_12h', []) if r['keep']]),
                    'n_significant_3d': len([r for r in task_data.get('ttest_report_3d', []) if r['keep']])
                }
        
        # Ajouter les résultats de régression
        for target, reg_data in regression_results.items():
            correlations = reg_data.get('correlations', [])
            selected_features = reg_data.get('selected_features', {})
            significant_corr = [r for r in correlations if r['significant_pearson']]
            
            global_significant['tasks'][target] = {
                'type': 'regression',
                'selected_features_12h': selected_features.get('12h', []),
                'selected_features_3d': selected_features.get('3d', []),
                'n_significant_correlations': len(significant_corr),
                'top_features': [r['feature'] for r in correlations[:10]]
            }
        
        global_file = Path(res_dir) / 'global_significant_features.json'
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(global_significant, f, indent=2, ensure_ascii=False)
        
        print(f"JSON global des features significatives sauvegardé dans : {global_file}")
    
    results["regression_analysis"] = regression_results
    return results


if __name__ == "__main__":
    # Exemple d’usage (adaptez les chemins à vos fichiers)
    DB_ROOT = "/home/ndecaux/NAS_EMPENN"
    actimetry = pd.read_excel("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_features.xlsx")
    participants = pd.read_excel("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx")
    # actimetry = pd.read_excel(".../actimetry_features.xlsx")
    # participants = pd.read_excel(".../participants_full_info.xlsx")
    acti_prefixed = actimetry.rename(columns=lambda col: f"acti_{col}" if col != "participant_id" else col)
    df = pd.merge(acti_prefixed, participants, on="participant_id", how="inner")
    #Drop participant_id sub-01025
    df = df[df['participant_id'] != 'sub-01025']
    
    results = main(df)
    # results = main_glm(df)
    with open("/home/ndecaux/Code/actiDep/analysis/results_acti_features.json", "w") as f:
        json.dump(results, f, indent=2)