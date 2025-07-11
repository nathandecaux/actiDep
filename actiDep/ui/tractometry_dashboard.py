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

    # Constantes (peuvent être des paramètres si besoin de flexibilité)
    model = 'staniz'
    pipeline = 'tractometry'

    def __init__(self, dataset_path: str = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids", **params):
        super().__init__(**params)
        self.dataset_path = dataset_path
        self.subjects_file_path = "/home/ndecaux/Code/actiDep/subjects.txt"
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

    def _load_data_for_current_metric(self):
        """
        Charge les données CSV pour la métrique actuellement sélectionnée (`self.metric`).
        Met à jour `self.df_all_bundles_for_metric`, `self.bundle_names_for_metric`,
        et les options/valeur de `self.selected_bundle`.
        """
        print(f"Chargement des données pour la métrique: {self.metric}...")
        
        pattern = opj(self.dataset_path, 'derivatives', self.pipeline, 'sub-*', 'metric', 
                     f'*_metric-{self.metric}_model-{self.model}_mean.csv')
        
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

    @param.depends('selected_bundle', 'metric', 'selected_subjects', 'group_by_column', 'show_individual_curves', 'show_group_averages') 
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
            parameters=['metric', 'selected_bundle', 'group_by_column', 'show_individual_curves', 'show_group_averages', 'filter_by_subjects_file'],
            widgets={
                'metric': {'type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 'options': ['FA', 'MD', 'RD', 'AD','IFW','IRF']},
                'selected_bundle': pn.widgets.Select,
                'group_by_column': pn.widgets.Select,
                'show_individual_curves': pn.widgets.Checkbox,
                'show_group_averages': pn.widgets.Checkbox,
                'filter_by_subjects_file': pn.widgets.Checkbox
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
            main=[plot_pane],
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
    viewer = SimpleBundleViewer() 
    
    try:
        app_view = viewer.view()
        
        print("Visualiseur simple prêt. Démarrage du serveur Panel...")
        # Utiliser un port différent si le port 5007 est déjà utilisé
        app_view.show(port=5008, open=True, title="Visualiseur de Faisceaux ActiDep") 
        
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
