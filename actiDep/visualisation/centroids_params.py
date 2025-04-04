"""
=================================================
Interface pour le clustering de tractogrammes
=================================================

Cette interface permet de:
1. Charger des images anatomiques et des tractogrammes
2. Appliquer le clustering QuickBundles avec différents paramètres
3. Visualiser interactivement les résultats
"""

import numpy as np
import os.path as op
from dipy.viz import actor, ui, window
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from fury.colormap import distinguishable_colormap
import os
os.environ['DIPY_VTK_BACKEND'] = 'offscreen'  # Essayez d'abord cette option


class TractoClusteringApp:
    def __init__(self):
        self.scene = window.Scene()
        self.streamlines = None
        self.data = None  # Données anatomiques (FA, T1, etc.)
        self.affine = None
        self.shape = None
        self.world_coords = True
        self.qb = None
        self.clusters = None
        self.centroids_actor = None
        self.image_actors = []
        self.slicer_opacity = 0.6
        self.threshold = 10.0  # Seuil par défaut pour QuickBundles
        self.nb_points = 12    # Nombre de points par défaut pour le resample
        
    def load_data(self, tractogram_file, anatomy_file=None):
        """Charge un tractogramme et optionnellement une image anatomique"""
        if op.exists(tractogram_file):
            # Charger le tractogramme
            sft = load_tractogram(tractogram_file, 'same', bbox_valid_check=False, trk_header_check=False)
            self.streamlines = Streamlines(sft.streamlines)
            self.affine = sft.affine
            print(f"Tractogramme chargé: {len(self.streamlines)} streamlines")
            
            # Charger l'image anatomique si fournie
            if anatomy_file and op.exists(anatomy_file):
                self.data, self.affine = load_nifti(anatomy_file)
                self.shape = self.data.shape
                print(f"Image anatomique chargée: {self.shape}")
        else:
            print(f"Fichier {tractogram_file} introuvable")
    
    def apply_clustering(self, threshold=None, nb_points=None):
        """Applique le clustering QuickBundles aux streamlines"""
        if self.streamlines is None:
            print("Aucune streamline à analyser")
            return
            
        if threshold is not None:
            self.threshold = threshold
        if nb_points is not None:
            self.nb_points = nb_points
            
        # Définir la métrique pour QuickBundles
        feature = ResampleFeature(nb_points=self.nb_points)
        metric = AveragePointwiseEuclideanMetric(feature=feature)
        
        # Créer l'instance QuickBundles et l'appliquer
        self.qb = QuickBundles(threshold=self.threshold, metric=metric)
        self.clusters = self.qb.cluster(self.streamlines)
        
        print(f"Clustering terminé: {len(self.clusters)} clusters trouvés")
        return self.clusters
    
    def _setup_anatomical_slices(self):
        """Configure les slicers pour l'image anatomique"""
        if self.data is None:
            return
            
        # Créer des slicers pour chaque plan
        for img_actor in self.image_actors:
            self.scene.rm(img_actor)
        self.image_actors = []
        
        if not self.world_coords:
            image_actor_z = actor.slicer(self.data, affine=np.eye(4))
        else:
            image_actor_z = actor.slicer(self.data, affine=self.affine)
        
        image_actor_z.opacity(self.slicer_opacity)
        
        # Ajouter des slicers pour X et Y
        image_actor_x = image_actor_z.copy()
        x_midpoint = int(np.round(self.shape[0] / 2))
        image_actor_x.display_extent(x_midpoint, x_midpoint, 0, self.shape[1] - 1, 0, self.shape[2] - 1)
        
        image_actor_y = image_actor_z.copy()
        y_midpoint = int(np.round(self.shape[1] / 2))
        image_actor_y.display_extent(0, self.shape[0] - 1, y_midpoint, y_midpoint, 0, self.shape[2] - 1)
        
        self.image_actors = [image_actor_z, image_actor_x, image_actor_y]
        for actor_obj in self.image_actors:
            self.scene.add(actor_obj)
    
    def visualize_clusters(self):
        """Visualise les clusters et les centroids dans la scène"""
        if self.clusters is None:
            print("Veuillez d'abord effectuer un clustering")
            return
            
        print('1')
        # Nettoyer la scène
        self.scene.clear()
        print('2')
        # Ajouter les slices anatomiques si disponibles
        if self.data is not None:
            print('3')
            self._setup_anatomical_slices()
        print('4')
        # Obtenir un jeu de couleurs distinctes
        max_colors = max(20, len(self.clusters))
        colors = []
        color_gen = distinguishable_colormap()
        for _ in range(max_colors):
            try:
                colors.append(next(color_gen))
            except StopIteration:
                # Si le générateur a moins de couleurs que nécessaire
                break
        print('5')
           
        # Afficher les centroids
        centroids = self.clusters.centroids
        if not self.world_coords and self.affine is not None:
            centroids = transform_streamlines(centroids, np.linalg.inv(self.affine))
            
        self.centroids_actor = actor.line(centroids, colors=colors[:len(centroids)], 
                                         linewidth=4.0)
        self.scene.add(self.centroids_actor)
        # Configurer l'interface utilisateur
        self._setup_ui()
        
        return self.show_manager
    
    def _setup_ui(self):
        """Configure les sliders et l'interface utilisateur"""
        if self.data is None:
            return
            
        # Créer le gestionnaire de fenêtre
        self.show_manager = window.ShowManager(scene=self.scene, size=(1200, 900))
        self.show_manager.initialize()
        
        # Créer des sliders pour X, Y, Z
        line_slider_z = ui.LineSlider2D(
            min_value=0,
            max_value=self.shape[2] - 1,
            initial_value=self.shape[2] / 2,
            text_template="{value:.0f}",
            length=140,
        )
        
        line_slider_x = ui.LineSlider2D(
            min_value=0,
            max_value=self.shape[0] - 1,
            initial_value=self.shape[0] / 2,
            text_template="{value:.0f}",
            length=140,
        )
        
        line_slider_y = ui.LineSlider2D(
            min_value=0,
            max_value=self.shape[1] - 1,
            initial_value=self.shape[1] / 2,
            text_template="{value:.0f}",
            length=140,
        )
        
        opacity_slider = ui.LineSlider2D(
            min_value=0.0, max_value=1.0, initial_value=self.slicer_opacity, length=140
        )
        
        # Slider pour le seuil de QuickBundles
        threshold_slider = ui.LineSlider2D(
            min_value=1.0, max_value=50.0, initial_value=self.threshold, length=140
        )
        
        # Slider pour le nombre de points dans le resampling
        nb_points_slider = ui.LineSlider2D(
            min_value=3, max_value=50, initial_value=self.nb_points, text_template="{value:.0f}", length=140
        )
        
        # Boîtes de texte éditables pour les paramètres de clustering
        threshold_textbox = ui.TextBox2D(height=30, width=50, text=str(self.threshold))
        nb_points_textbox = ui.TextBox2D(height=30, width=50, text=str(self.nb_points))
        
        # Callbacks pour les sliders
        def change_slice_z(slider):
            z = int(np.round(slider.value))
            self.image_actors[0].display_extent(0, self.shape[0] - 1, 0, self.shape[1] - 1, z, z)
            
        def change_slice_x(slider):
            x = int(np.round(slider.value))
            self.image_actors[1].display_extent(x, x, 0, self.shape[1] - 1, 0, self.shape[2] - 1)
            
        def change_slice_y(slider):
            y = int(np.round(slider.value))
            self.image_actors[2].display_extent(0, self.shape[0] - 1, y, y, 0, self.shape[2] - 1)
            
        def change_opacity(slider):
            opacity = slider.value
            for img_actor in self.image_actors:
                img_actor.opacity(opacity)
                
        def change_threshold(slider):
            new_value = slider.value
            self.threshold = new_value
            # Mettre à jour la boîte de texte
            threshold_textbox.text = f"{new_value:.2f}"
            self.apply_clustering()
            self.update_centroids_visualization()
            
        def change_nb_points(slider):
            new_value = int(slider.value)
            self.nb_points = new_value
            # Mettre à jour la boîte de texte
            nb_points_textbox.text = str(new_value)
            self.apply_clustering()
            self.update_centroids_visualization()
            
        def update_threshold_from_textbox(textbox):
            try:
                new_value = float(textbox.text)
                if 1.0 <= new_value <= 50.0:
                    self.threshold = new_value
                    # Mettre à jour le slider
                    threshold_slider.set_value(new_value)
                    self.apply_clustering()
                    self.update_centroids_visualization()
                else:
                    # Réinitialiser au seuil actuel si hors limites
                    textbox.text = f"{self.threshold:.2f}"
            except ValueError:
                # Réinitialiser au seuil actuel si non numérique
                textbox.text = f"{self.threshold:.2f}"
                
        def update_nb_points_from_textbox(textbox):
            try:
                new_value = int(textbox.text)
                if 3 <= new_value <= 50:
                    self.nb_points = new_value
                    # Mettre à jour le slider
                    nb_points_slider.set_value(new_value)
                    self.apply_clustering()
                    self.update_centroids_visualization()
                else:
                    # Réinitialiser au nombre actuel si hors limites
                    textbox.text = str(self.nb_points)
            except ValueError:
                # Réinitialiser au nombre actuel si non numérique
                textbox.text = str(self.nb_points)
            
        line_slider_z.on_change = change_slice_z
        line_slider_x.on_change = change_slice_x
        line_slider_y.on_change = change_slice_y
        opacity_slider.on_change = change_opacity
        threshold_slider.on_change = change_threshold
        nb_points_slider.on_change = change_nb_points
        
        # Lier les callbacks pour les boîtes de texte
        threshold_textbox.on_submit = update_threshold_from_textbox
        nb_points_textbox.on_submit = update_nb_points_from_textbox
        
        # Créer les étiquettes
        def build_label(text):
            label = ui.TextBlock2D()
            label.message = text
            label.font_size = 18
            label.font_family = "Arial"
            label.justification = "left"
            label.bold = False
            label.italic = False
            label.shadow = False
            label.background_color = (0, 0, 0)
            label.color = (1, 1, 1)
            return label
            
        line_slider_label_z = build_label(text="Z Slice")
        line_slider_label_x = build_label(text="X Slice")
        line_slider_label_y = build_label(text="Y Slice")
        opacity_slider_label = build_label(text="Opacity")
        threshold_slider_label = build_label(text="QB Threshold")
        nb_points_slider_label = build_label(text="Resample Points")
        
        # Créer un panneau plus grand pour les contrôles
        panel = ui.Panel2D(size=(350, 400), color=(1, 1, 1), opacity=0.2, align="right")
        panel.center = (1030, 200)
        
        # Ajouter les éléments au panneau
        # Première colonne : étiquettes
        panel.add_element(line_slider_label_x, (0.05, 0.90))
        panel.add_element(line_slider_label_y, (0.05, 0.80))
        panel.add_element(line_slider_label_z, (0.05, 0.70))
        panel.add_element(opacity_slider_label, (0.05, 0.60))
        panel.add_element(threshold_slider_label, (0.05, 0.40))
        panel.add_element(nb_points_slider_label, (0.05, 0.20))
        
        # Deuxième colonne : sliders
        panel.add_element(line_slider_x, (0.38, 0.90))
        panel.add_element(line_slider_y, (0.38, 0.80))
        panel.add_element(line_slider_z, (0.38, 0.70))
        panel.add_element(opacity_slider, (0.38, 0.60))
        panel.add_element(threshold_slider, (0.38, 0.40))
        panel.add_element(nb_points_slider, (0.38, 0.20))
        
        # Ajouter les boîtes de texte éditables
        panel.add_element(threshold_textbox, (0.80, 0.40))
        panel.add_element(nb_points_textbox, (0.80, 0.20))
        
        # Ajouter un titre au panneau
        title_label = build_label(text="Paramètres de clustering")
        title_label.font_size = 20
        title_label.bold = True
        panel.add_element(title_label, (0.05, 0.95))
        
        # Étiquette explicative
        info_label = build_label(text="Modifiez les valeurs via le slider\nou en saisissant directement")
        info_label.font_size = 14
        panel.add_element(info_label, (0.10, 0.50))
        
        self.scene.add(panel)
        
        # Callback pour ajuster le panneau lors du redimensionnement de la fenêtre
        global size
        size = self.scene.GetSize()
        
        def win_callback(obj, event):
            global size
            if size != obj.GetSize():
                size_old = size
                size = obj.GetSize()
                size_change = [size[0] - size_old[0], 0]
                panel.re_align(size_change)
                
        self.show_manager.add_window_callback(win_callback)
        
    def update_centroids_visualization(self):
        """Met à jour la visualisation des centroids après modification des paramètres"""
        if self.centroids_actor is not None:
            self.scene.rm(self.centroids_actor)
            
        if self.clusters is not None:
            max_colors = max(20, len(self.clusters))
            colors = []
            color_gen = distinguishable_colormap()
            for _ in range(max_colors):
                try:
                    colors.append(next(color_gen))
                except StopIteration:
                    # Si le générateur a moins de couleurs que nécessaire
                    break 
            centroids = self.clusters.centroids
            
            if not self.world_coords and self.affine is not None:
                centroids = transform_streamlines(centroids, np.linalg.inv(self.affine))
                
            self.centroids_actor = actor.line(centroids, colors=colors[:len(centroids)], 
                                             linewidth=4.0)
            self.scene.add(self.centroids_actor)
        
    def start(self):
        """Démarre l'interface de visualisation interactive"""
        if hasattr(self, 'show_manager'):
            self.scene.zoom(1.5)
            self.scene.reset_clipping_range()
            self.show_manager.render()
            self.show_manager.start()
        else:
            print("Visualisez d'abord les clusters avec visualize_clusters()")
            
    def capture_screenshot(self, filename):
        """Capture une image de la scène actuelle"""
        if hasattr(self, 'show_manager'):
            window.record(scene=self.scene, out_path=filename, size=(1200, 900))
            print(f"Capture enregistrée sous {filename}")


if __name__ == "__main__":
    # Activer le débogage VTK
    import os
    os.environ['VTK_DEBUG_LEAKS'] = '1'  # Active le débogage VTK pour plus d'informations    
    # Créer l'application
    app = TractoClusteringApp()
    
    # Vous devez spécifier vos propres fichiers ici
    tractogram_file = "sub-03011_desc-normalized_label-WM_model-fod_tracto.trk"
    anatomy_file = "sub-03011_space-B0_T1w.nii.gz"
    
    # Charger les données
    app.load_data(tractogram_file, anatomy_file)
    
    # Appliquer le clustering initial
    app.apply_clustering(threshold=10.0, nb_points=12)
    
    # Essayez d'abord une capture d'écran pour vérifier le rendu
    print('visualisation')
    app.visualize_clusters()
    print('capture')

    
    # Puis essayez l'affichage interactif
    app.start()