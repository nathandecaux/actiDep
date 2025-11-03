import numpy as np
import dipy
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
from os.path import join as opj
import os
import imageio.v2 as imageio
import argparse

def setup_scene(bundle_file, anat_file):
    """Configurer une scène avec un bundle et l'anatomie."""
    # Charger les données anatomiques
    data, affine = load_nifti(anat_file)

    # Charger le bundle
    bundle_obj = load_trk(bundle_file, reference=anat_file)
    streamlines = bundle_obj.streamlines

    # Créer la scène
    scene = window.Scene()
    scene.background((0.5, 0.5, 0.5))  # Fond gris

    # Calculer la plage de valeurs pour l'image anatomique
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 2 * std)

    # Ajouter les coupes anatomiques
    slice_actor = actor.slicer(data, affine=affine, value_range=value_range)

    # Obtenir les dimensions de l'image
    shape = data.shape

    # Configurer les positions des coupes (milieu par défaut)
    x_midpoint = int(np.round(shape[0] / 2))
    y_midpoint = int(np.round(shape[1] / 2))
    z_midpoint = int(np.round(shape[2] / 2))

    # Configurer les coupes selon les 3 axes
    slice_actor_x = slice_actor.copy()
    slice_actor_x.display(x=x_midpoint)

    slice_actor_y = slice_actor.copy()
    slice_actor_y.display(y=y_midpoint)

    slice_actor_z = slice_actor.copy()
    slice_actor_z.display(z=z_midpoint)

    # Configurer la transparence des coupes
    slice_opacity = 0.6
    slice_actor_x.opacity(slice_opacity)
    slice_actor_y.opacity(slice_opacity)
    slice_actor_z.opacity(slice_opacity)

    # Créer l'acteur pour le bundle avec une couleur spécifique (peut être amélioré)
    # Utilisation d'une couleur fixe pour simplifier le script
    stream_actor = actor.line(streamlines, colors=(1, 0, 0), linewidth=2)

    # Ajouter tous les acteurs à la scène
    scene.add(stream_actor)
    scene.add(slice_actor_x)
    scene.add(slice_actor_y)
    scene.add(slice_actor_z)

    # Ajuster la caméra
    scene.reset_camera()
    scene.zoom(1.5)

    return scene

def create_rotating_gif(scene, output_path, size=(800, 800), n_frames=36, fps=10):
    """Créer un GIF avec une vue rotative d'une scène FURY."""
    # Générer les frames pour le GIF
    frames = []
    angle_step = 360 // n_frames
    for _ in range(n_frames):
        # Faire tourner la caméra autour de la scène
        scene.azimuth(angle_step)
        # Mettre à jour la scène (important pour que la rotation soit appliquée)
        frame = window.snapshot(scene, size=size, offscreen=True) # Utiliser offscreen=True
        frames.append(frame)

    # Sauvegarder le GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF sauvegardé : {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Générer des GIFs rotatifs de faisceaux de fibres.")
    parser.add_argument("bundle_dir", help="Répertoire contenant les fichiers de faisceaux (.trk).")
    parser.add_argument("anat_file", help="Chemin vers le fichier anatomique NIfTI (.nii.gz).")
    parser.add_argument("output_dir", help="Répertoire où sauvegarder les GIFs générés.")
    parser.add_argument("--num_bundles", type=int, default=None, help="Nombre maximum de faisceaux à traiter (optionnel).")
    parser.add_argument("--size", type=int, nargs=2, default=[800, 800], help="Taille du GIF (largeur hauteur).")
    parser.add_argument("--n_frames", type=int, default=36, help="Nombre d'images dans le GIF.")
    parser.add_argument("--fps", type=int, default=10, help="Images par seconde pour le GIF.")

    args = parser.parse_args()

    # Vérifier l'existence du fichier anatomique
    if not os.path.isfile(args.anat_file):
        print(f"Erreur : Le fichier anatomique '{args.anat_file}' n'a pas été trouvé.")
        return

    # Trouver les fichiers de faisceaux
    try:
        bundle_files = [f for f in os.listdir(args.bundle_dir) if f.endswith('.trk')]
        if not bundle_files:
            print(f"Erreur : Aucun fichier .trk trouvé dans '{args.bundle_dir}'.")
            return
        bundle_files = [opj(args.bundle_dir, f) for f in bundle_files]
    except FileNotFoundError:
        print(f"Erreur : Le répertoire des faisceaux '{args.bundle_dir}' n'a pas été trouvé.")
        return

    # Limiter le nombre de faisceaux si demandé
    if args.num_bundles is not None and args.num_bundles > 0:
        bundle_files = bundle_files[:args.num_bundles]

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Début de la génération des GIFs pour {len(bundle_files)} faisceaux...")

    # Traiter chaque faisceau
    for bundle_file in bundle_files:
        bundle_name = os.path.splitext(os.path.basename(bundle_file))[0]
        print(f"Traitement de {bundle_name}...")

        try:
            # Configurer la scène
            scene = setup_scene(bundle_file, args.anat_file)

            # Définir le chemin de sortie pour le GIF
            output_gif_path = opj(args.output_dir, f"{bundle_name}_rotating.gif")

            # Créer le GIF
            create_rotating_gif(scene, output_gif_path, size=tuple(args.size), n_frames=args.n_frames, fps=args.fps)

            # Fermer la fenêtre FURY associée à la scène pour libérer les ressources
            # (Important lors du traitement de plusieurs fichiers en boucle)
            scene.clear() # Nettoyer les acteurs
            # Malheureusement, FURY/VTK peut avoir des problèmes de gestion de mémoire/fenêtre en mode offscreen pur.
            # Si des problèmes persistent, envisagez d'exécuter chaque GIF dans un processus séparé.

        except Exception as e:
            print(f"Erreur lors du traitement de {bundle_name}: {e}")
            # Continuer avec le fichier suivant

    print(f"Traitement terminé. Les GIFs sont sauvegardés dans {args.output_dir}")

if __name__ == "__main__":
    bundle_dir = '/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/'
    atlas_dir = '/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/'

    atlas_anat = opj(atlas_dir, 'average_anat.nii.gz')

    #Call the main function with the arguments
    argparse_args = argparse.Namespace(
        bundle_dir=bundle_dir,
        anat_file=atlas_anat,
        output_dir='/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/rotating_gifs/',
        num_bundles=None,
        size=[800, 800],
        n_frames=36,
        fps=10
    )

    # Call the main function with the arguments
    main_args = argparse_args
    main_args.bundle_dir = bundle_dir
    main_args.anat_file = atlas_anat
    main_args.output_dir = 'Atlas/rotating_gifs/'
    main_args.num_bundles = None
    main_args.size = [800, 800]
    main_args.n_frames = 36
    main_args.fps = 10
    main_args = argparse.Namespace(**vars(main_args))
    main()

    
