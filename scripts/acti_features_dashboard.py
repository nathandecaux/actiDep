import pandas as pd
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts
import param

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from scipy import stats
from scipy.stats import ttest_ind, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration Panel et HoloViews
pn.extension('tabulator', 'bokeh')
hv.extension('bokeh')

# Importer la fonction de résidualisation depuis analyze_acti_features.py
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

class ActiFeaturesDashboard(param.Parameterized):
    """Dashboard pour l'analyse des features d'activité"""
    
    # Paramètres de sélection
    selected_features = param.List(default=[], doc="Features sélectionnées")
    target_type = param.Selector(default="classification", 
                                objects=["classification", "regression", "correlation"])
    target_variable = param.Selector(default="group", 
                                   objects=["group", "apathy", "ami", "aes"])
    
    def __init__(self, dataframe, **params):
        super().__init__(**params)
        self.df = dataframe.copy()
        
        # Filtrer seulement les colonnes qui commencent par 'acti_'
        self.feature_columns = [col for col in self.df.columns if col.startswith("acti_")]
        
        # Vérifier les colonnes target disponibles dans le dataframe
        available_targets = []
        for target in ['group', 'apathy', 'ami', 'aes']:
            if target in self.df.columns:
                available_targets.append(target)
        
        if not available_targets:
            raise ValueError("Aucune colonne target trouvée (group, apathy, ami, aes)")
        
        # Initialiser les paramètres
        self.param.target_variable.objects = available_targets
        if available_targets:
            self.target_variable = available_targets[0]
    
    def get_clean_data(self):
        """Retourne les données nettoyées pour l'analyse"""
        if not self.selected_features:
            return pd.DataFrame()
        
        # Sélection des colonnes
        cols = self.selected_features + [self.target_variable]
        data = self.df[cols].copy()
        
        # Suppression des valeurs manquantes
        data = data.dropna()
        
        return data
    
    def get_corrected_data(self):
        """Retourne les données avec correction des facteurs confondants"""
        data = self.get_clean_data()
        if data.empty:
            return pd.DataFrame()
        
        # Variables de confusion disponibles
        confounders = []
        if self.target_variable == 'group':
            potential_confounders = ["age", "sex", "city"]
        else:
            potential_confounders = ["age", "sex", "city", 'duration_dep', 'type_dep']
        
        confounders = [c for c in potential_confounders if c in self.df.columns]
        
        if not confounders:
            return data
        
        corrected_data = data.copy()
        
        # Correction de chaque feature
        for feature in self.selected_features:
            if feature in corrected_data.columns:
                corrected_data[feature] = ols_residualize(
                    self.df[feature], 
                    self.df[confounders]
                )
        
        return corrected_data.dropna()

    @pn.depends('selected_features', 'target_type', 'target_variable', watch=True)
    def view_distributions(self):
        """Calcule et affiche les distributions des features"""
        data = self.get_clean_data()
        if data.empty:
            return pn.pane.HTML("<p>Sélectionnez des features pour voir les distributions</p>")
        
        try:
            plots = []
            
            for feature in self.selected_features[:8]:  # Augmenter à 8 features
                if self.target_type == "classification":
                    # Boxplots par groupe pour classification
                    groups = data[self.target_variable].unique()
                    box_data = []
                    
                    for group in groups:
                        subset = data[data[self.target_variable] == group][feature]
                        if len(subset) > 0:
                            # Créer les données pour boxplot
                            for value in subset.values:
                                box_data.append((str(group), value))
                    
                    if box_data:
                        plot = hv.BoxWhisker(
                            box_data, 
                            kdims=['Groupe'], 
                            vdims=['Valeur']
                        ).opts(
                            title=f"{feature}",
                            width=450, height=350,
                            xlabel="Groupe", ylabel="Valeur",
                            box_color=hv.Cycle(['lightblue', 'lightcoral', 'lightgreen', 'gold']),
                            fontsize={'title': 12, 'labels': 10, 'xticks': 9, 'yticks': 9},
                            xrotation=45
                        )
                    else:
                        plot = hv.Text(0.5, 0.5, f"Pas de données\npour {feature}").opts(
                            width=450, height=350
                        )
                else:
                    # Boxplot simple pour régression/corrélation
                    values = data[feature].dropna()
                    if len(values) > 0:
                        plot = hv.BoxWhisker(
                            [(feature, val) for val in values],
                            kdims=['Feature'], 
                            vdims=['Valeur']
                        ).opts(
                            title=f"{feature}",
                            width=450, height=350,
                            xlabel="", ylabel="Valeur",
                            box_color='lightblue',
                            fontsize={'title': 12, 'labels': 10, 'xticks': 9, 'yticks': 9}
                        )
                    else:
                        plot = hv.Text(0.5, 0.5, f"Pas de données\npour {feature}").opts(
                            width=450, height=350
                        )
                
                plots.append(plot)
            
            if len(plots) == 1:
                layout = plots[0]
            elif len(plots) <= 4:
                layout = hv.Layout(plots).cols(2)
            else:
                layout = hv.Layout(plots).cols(2)  # Garder 2 colonnes même pour plus de plots
            
            return pn.pane.HoloViews(layout, sizing_mode='stretch_width')
            
        except Exception as e:
            return pn.pane.HTML(f"<p>Erreur lors du calcul des distributions: {str(e)}</p>")

    @pn.depends('selected_features', 'target_type', 'target_variable', watch=True)
    def view_corrected_distributions(self):
        """Calcule et affiche les distributions corrigées"""
        data = self.get_corrected_data()
        if data.empty:
            return pn.pane.HTML("<p>Pas de données corrigées disponibles</p>")
        
        try:
            plots = []
            
            for feature in self.selected_features[:8]:  # Augmenter à 8 features
                if feature not in data.columns:
                    continue
                    
                if self.target_type == "classification":
                    # Boxplots par groupe pour classification
                    groups = data[self.target_variable].unique()
                    box_data = []
                    
                    for group in groups:
                        subset = data[data[self.target_variable] == group][feature]
                        if len(subset) > 0:
                            # Créer les données pour boxplot
                            for value in subset.values:
                                box_data.append((str(group), value))
                    
                    if box_data:
                        plot = hv.BoxWhisker(
                            box_data, 
                            kdims=['Groupe'], 
                            vdims=['Valeur corrigée']
                        ).opts(
                            title=f"{feature} (corrigé)",
                            width=450, height=350,
                            xlabel="Groupe", ylabel="Valeur corrigée",
                            box_color=hv.Cycle(['lightblue', 'lightcoral', 'lightgreen', 'gold']),
                            fontsize={'title': 12, 'labels': 10, 'xticks': 9, 'yticks': 9},
                            xrotation=45
                        )
                    else:
                        plot = hv.Text(0.5, 0.5, f"Pas de données\npour {feature}").opts(
                            width=450, height=350
                        )
                else:
                    # Boxplot simple pour régression/corrélation
                    values = data[feature].dropna()
                    if len(values) > 0:
                        plot = hv.BoxWhisker(
                            [(feature, val) for val in values],
                            kdims=['Feature'], 
                            vdims=['Valeur corrigée']
                        ).opts(
                            title=f"{feature} (corrigé)",
                            width=450, height=350,
                            xlabel="", ylabel="Valeur corrigée",
                            box_color='lightgreen',
                            fontsize={'title': 12, 'labels': 10, 'xticks': 9, 'yticks': 9}
                        )
                    else:
                        plot = hv.Text(0.5, 0.5, f"Pas de données\npour {feature}").opts(
                            width=450, height=350
                        )
                
                plots.append(plot)
            
            if len(plots) == 1:
                layout = plots[0]
            elif len(plots) > 1:
                layout = hv.Layout(plots).cols(2)
            else:
                return pn.pane.HTML("<p>Aucune donnée corrigée disponible</p>")
            
            return pn.pane.HoloViews(layout, sizing_mode='stretch_width')
            
        except Exception as e:
            return pn.pane.HTML(f"<p>Erreur lors du calcul des distributions corrigées: {str(e)}</p>")

    @pn.depends('selected_features', 'target_type', 'target_variable', watch=True) 
    def view_statistical_tests(self):
        """Calcule et affiche les tests statistiques"""
        data = self.get_clean_data()
        if data.empty:
            return pn.pane.HTML("<p>Sélectionnez des features</p>")
        
        results = []
        
        for feature in self.selected_features:
            try:
                if self.target_type == "classification":
                    # Tests pour classification
                    groups = data[self.target_variable].unique()
                    if len(groups) == 2:
                        group1_data = data[data[self.target_variable] == groups[0]][feature]
                        group2_data = data[data[self.target_variable] == groups[1]][feature]
                        
                        # T-test
                        t_stat, t_pval = ttest_ind(group1_data, group2_data)
                        
                        # Mann-Whitney U test
                        u_stat, u_pval = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        
                        results.extend([
                            {
                                'Feature': feature,
                                'Test': 'T-test',
                                'Statistique': f"{t_stat:.4f}",
                                'P-value': f"{t_pval:.4f}",
                                'Significatif': "Oui" if t_pval < 0.05 else "Non"
                            },
                            {
                                'Feature': feature,
                                'Test': 'Mann-Whitney U',
                                'Statistique': f"{u_stat:.4f}",
                                'P-value': f"{u_pval:.4f}",
                                'Significatif': "Oui" if u_pval < 0.05 else "Non"
                            }
                        ])
                else:
                    # Tests de corrélation
                    target_data = data[self.target_variable]
                    feature_data = data[feature]
                    
                    # Corrélation de Pearson
                    pearson_r, pearson_p = pearsonr(feature_data, target_data)
                    
                    # Corrélation de Spearman
                    spearman_r, spearman_p = spearmanr(feature_data, target_data)
                    
                    results.extend([
                        {
                            'Feature': feature,
                            'Test': 'Pearson',
                            'Corrélation': f"{pearson_r:.4f}",
                            'P-value': f"{pearson_p:.4f}",
                            'Significatif': "Oui" if pearson_p < 0.05 else "Non"
                        },
                        {
                            'Feature': feature,
                            'Test': 'Spearman',
                            'Corrélation': f"{spearman_r:.4f}",
                            'P-value': f"{spearman_p:.4f}",
                            'Significatif': "Oui" if spearman_p < 0.05 else "Non"
                        }
                    ])
            except Exception as e:
                results.append({
                    'Feature': feature,
                    'Test': 'Erreur',
                    'Statistique': str(e)[:30],
                    'P-value': 'N/A',
                    'Significatif': 'N/A'
                })
        
        if results:
            df_results = pd.DataFrame(results)
            return pn.widgets.Tabulator(
                df_results, 
                pagination='remote', 
                page_size=20, 
                height=500,
                width=800,
                show_index=False
            )
        else:
            return pn.pane.HTML("<p>Aucun résultat disponible</p>")

    @pn.depends('selected_features', 'target_type', 'target_variable', watch=True)
    def view_ml_performance(self):
        """Calcule et affiche les performances ML"""
        data = self.get_clean_data()
        if data.empty or len(self.selected_features) < 1:
            return pn.pane.HTML("<p>Sélectionnez des features</p>")
        
        X = data[self.selected_features]
        y = data[self.target_variable]
        
        # Vérifier qu'on a assez de données
        if len(X) < 10:
            return pn.pane.HTML("<p>Pas assez de données pour l'évaluation ML</p>")
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = []
        
        if self.target_type == "classification":
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
            scoring = 'accuracy'
            
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR()
            }
            
            cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
            scoring = 'r2'
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                results.append({
                    'Modèle': model_name,
                    'Score moyen': f"{scores.mean():.4f}",
                    'Écart-type': f"{scores.std():.4f}",
                    'Score min': f"{scores.min():.4f}",
                    'Score max': f"{scores.max():.4f}"
                })
            except Exception as e:
                results.append({
                    'Modèle': model_name,
                    'Score moyen': f"Erreur: {str(e)[:30]}",
                    'Écart-type': '-',
                    'Score min': '-',
                    'Score max': '-'
                })
        
        if results:
            df_results = pd.DataFrame(results)
            return pn.widgets.Tabulator(
                df_results, 
                height=350,
                width=800,
                show_index=False
            )
        else:
            return pn.pane.HTML("<p>Aucun résultat ML disponible</p>")

    def create_dashboard(self):
        """Crée le dashboard complet"""
        
        # Widgets de contrôle - Remplacer CrossSelector par MultiSelect
        feature_selector = pn.widgets.MultiSelect(
            name="Sélection des Features",
            value=[],
            options=self.feature_columns,
            height=400,
            width=380,
            size=15  # Nombre de lignes visibles
        )
        
        target_type_selector = pn.widgets.Select(
            name="Type d'analyse",
            options=["classification", "regression", "correlation"],
            value="classification",
            width=380
        )
        
        available_targets = [t for t in ['group', 'apathy', 'ami', 'aes'] if t in self.df.columns]
        target_selector = pn.widgets.Select(
            name="Variable target",
            options=available_targets,
            value=available_targets[0] if available_targets else "group",
            width=380
        )
        
        # Liaison bidirectionnelle avec les paramètres - Conserver les valeurs lors des mises à jour
        def update_features(new_features):
            old_features = self.selected_features
            self.selected_features = new_features
            # Ne pas réinitialiser le widget si les valeurs sont identiques
            if old_features != new_features:
                print(f"Features mises à jour: {len(new_features)} sélectionnées")
        
        def update_target_type(new_type):
            old_type = self.target_type
            self.target_type = new_type
            if old_type != new_type:
                print(f"Type d'analyse mis à jour: {new_type}")
        
        def update_target_variable(new_target):
            old_target = self.target_variable
            self.target_variable = new_target
            if old_target != new_target:
                print(f"Variable target mise à jour: {new_target}")
        
        # Utiliser pn.bind avec watch=True pour une synchronisation stable
        pn.bind(update_features, feature_selector.param.value, watch=True)
        pn.bind(update_target_type, target_type_selector.param.value, watch=True)
        pn.bind(update_target_variable, target_selector.param.value, watch=True)
        
        # Bouton pour sélectionner/désélectionner toutes les features
        select_all_button = pn.widgets.Button(
            name="Sélectionner tout", 
            button_type="primary",
            width=185
        )
        
        clear_all_button = pn.widgets.Button(
            name="Tout désélectionner", 
            button_type="primary",
            width=185
        )
        
        def select_all_features(event):
            feature_selector.value = self.feature_columns[:20]  # Limiter à 20 pour performance
        
        def clear_all_features(event):
            feature_selector.value = []
        
        select_all_button.on_click(select_all_features)
        clear_all_button.on_click(clear_all_features)
        
        # Widget de recherche pour filtrer les features
        search_input = pn.widgets.TextInput(
            name="Rechercher features",
            placeholder="Tapez pour filtrer...",
            width=380
        )
        
        def filter_features(event):
            search_term = event.new.lower()
            if search_term:
                filtered_options = [col for col in self.feature_columns if search_term in col.lower()]
            else:
                filtered_options = self.feature_columns
            feature_selector.options = filtered_options
        
        search_input.param.watch(filter_features, 'value')
        
        # Interface de contrôle améliorée
        controls = pn.Column(
            "## Paramètres d'analyse",
            search_input,
            feature_selector,
            pn.Row(select_all_button, clear_all_button),
            "---",
            target_type_selector,
            target_selector,
            "---",
            f"**Features disponibles:** {len(self.feature_columns)}",
            pn.pane.HTML("<small>Utilisez Ctrl+clic pour sélections multiples</small>"),
            width=400,
            height=650
        )
        
        # Onglets de résultats
        tabs = pn.Tabs(
            ("Distributions originales", self.view_distributions),
            ("Distributions corrigées", self.view_corrected_distributions),
            ("Tests statistiques", self.view_statistical_tests),
            ("Performance ML", self.view_ml_performance),
            dynamic=True,
            tabs_location='above'
        )
        
        # Layout principal avec plus d'espace
        main_layout = pn.Row(
            controls,
            pn.Column(
                "## Résultats d'analyse", 
                tabs, 
                width=1000,  # Augmenter la largeur des résultats
                height=700   # Augmenter la hauteur
            ),
            sizing_mode='stretch_width'
        )
        
        return pn.template.FastListTemplate(
            title="Dashboard d'analyse des features d'activité", 
            main=[main_layout],
            header_background='#2596be',
            sidebar_width=420
        )

def create_dashboard(dataframe):
    """Fonction principale pour créer le dashboard"""
    dashboard_obj = ActiFeaturesDashboard(dataframe)
    return dashboard_obj.create_dashboard()

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des données comme dans analyze_acti_features.py
    actimetry = pd.read_excel("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/actimetry_features.xlsx")
    participants = pd.read_excel("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/participants_full_info.xlsx")
    
    # Préfixage des colonnes d'actimétrie comme dans analyze_acti_features.py
    acti_prefixed = actimetry.rename(columns=lambda col: f"acti_{col}" if col != "participant_id" else col)
    df = pd.merge(acti_prefixed, participants, on="participant_id", how="inner")
    #Drop participant_id == sub-01025
    df = df[df['participant_id'] != 'sub-01025']
    print(f"Colonnes disponibles commençant par 'acti_': {[c for c in df.columns if c.startswith('acti_')][:10]}...")
    print(f"Colonnes target disponibles: {[c for c in df.columns if c in ['group', 'apathy', 'ami', 'aes']]}")
    
    # Création du dashboard
    dashboard = create_dashboard(df)
    dashboard.servable()
    dashboard.show(port=5007)
    
    print("Dashboard créé. Utilisez create_dashboard(df) avec votre DataFrame.")
