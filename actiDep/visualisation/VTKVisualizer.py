from __future__ import annotations
from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence, Tuple
import os
import warnings
import math
import numpy as np
import contextlib
import pyvista as pv
import subprocess
import shutil
import tempfile

try:
    import imageio.v2 as imageio  # GIF option
except Exception:  # pragma: no cover
    imageio = None


DataInput = Union[str, Path, pv.DataSet]
ParamDict = Dict[str, object]


class VTKVisualizer:
    """
    Visualisation simplifiée de fichiers/datasets VTK via pyvista.

    Paramètres par dataset (clé -> signification):
      - display_array: str | None  nom de l'array scalaires à afficher
      - cmap: str | Sequence[str]  colormap matplotlib
      - clim: Tuple[float,float]   limites (min, max) pour l'affichage
      - threshold: (str, (min,max))  filtrage sur un array => applique un threshold
      - opacity: float | str | Sequence[float]
      - show_edges: bool
      - scalar_bar: bool
      - point_size / line_width: tailles optionnelles
      - ambient / specular / diffuse: réglages matériaux
      - style: 'surface' | 'wireframe' | 'points'   (nouveau; défaut surface)
      - render_points_as_spheres: bool (utile si style='points')
      - name: nom interne (sinon auto)
    Paramètres globaux (constructeur):
      - background: couleur du fond (default "white")
      - off_screen: bool (pour rendu sans interface, ex: serveur)
    """
    def __init__(self, background: str = "white", off_screen: bool = False):
        self._datasets: List[Tuple[pv.DataSet, ParamDict]] = []
        self.background = background
        self.off_screen = off_screen
        self._plotter: Optional[pv.Plotter] = None
        self._scalar_bar_added = False
        self._font_color = self._choose_font_color(self.background)
        # Auto bascule headless si pas de DISPLAY
        if not self.off_screen and not os.environ.get("DISPLAY"):
            warnings.warn("Aucun DISPLAY détecté -> passage en mode off_screen.")
            self.off_screen = True

    # ------------------------------
    # Chargement / ajout de datasets
    # ------------------------------
    @staticmethod
    def _load(obj: DataInput) -> pv.DataSet:
        if isinstance(obj, pv.DataSet):
            return obj
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(path)
        return pv.read(path)

    def add_dataset(self, data: DataInput, params: Optional[ParamDict] = None):
        ds = self._load(data)
        params = dict(params or {})
        if "name" not in params:
            params["name"] = f"ds{len(self._datasets)}"
        self._datasets.append((ds, params))
        # Invalidation plotter si déjà construit
        self._plotter = None
        return self

    @classmethod
    def from_paths(cls, paths: Sequence[DataInput], params_list: Optional[Sequence[ParamDict]] = None, **kwargs):
        vis = cls(**kwargs)
        params_list = params_list or [{}] * len(paths)
        for p, prm in zip(paths, params_list):
            vis.add_dataset(p, prm)
        return vis

    # ------------------------------
    # Construction de la scène
    # ------------------------------
    def _ensure_plotter(self):
        if self._plotter is not None:
            return
        self._plotter = pv.Plotter(off_screen=self.off_screen)
        self._plotter.set_background(self.background)
        # Recalcule (au cas où background modifié avant nouvel ensure)
        self._font_color = self._choose_font_color(self.background)
        self._scalar_bar_added = False
        for ds, prm in self._datasets:
            mesh = ds
            # Threshold si demandé
            if "threshold" in prm and prm["threshold"]:
                arr_name, (vmin, vmax) = prm["threshold"]
                if arr_name not in mesh.array_names:
                    raise ValueError(f"Array '{arr_name}' introuvable pour threshold.")
                mesh = mesh.threshold(value=(vmin, vmax), scalars=arr_name, invert=False)

            display_array = prm.get("display_array")
            if display_array and display_array not in mesh.array_names:
                raise ValueError(f"Array '{display_array}' introuvable dans dataset ({mesh.array_names}).")
            if display_array:
                # Assure que l'array est active (facilite la génération correcte de la scalar_bar)
                try:
                    mesh.set_active_scalars(display_array)
                except Exception:
                    pass

            # Mapping des paramètres vers add_mesh
            add_kwargs = dict(
                scalars=display_array,
                cmap=prm.get("cmap"),
                clim=prm.get("clim"),
                opacity=prm.get("opacity", 1.0),
                show_edges=prm.get("show_edges", False),
                # Une seule scalar bar globale (premier dataset éligible)
                show_scalar_bar=bool(display_array) and prm.get("scalar_bar", True) and not self._scalar_bar_added,
                name=prm.get("name"),
                color=prm.get("color"),
                style=prm.get("style"),  # 'surface' | 'wireframe' | 'points'
            )
            # Si rendu en points sans point_size défini, appliquer une valeur par défaut
            if add_kwargs.get("style") == "points" and "point_size" not in prm:
                add_kwargs["point_size"] = 5
            # Option de rendu sphérique des points
            if "render_points_as_spheres" in prm:
                add_kwargs["render_points_as_spheres"] = prm["render_points_as_spheres"]
            if add_kwargs["show_scalar_bar"]:
                sb_args = prm.get("scalar_bar_args") or {}
                sb_defaults = {
                    "title": display_array or "",
                    "n_labels": 5,
                    "fmt": "%.2f",
                    # Couleur de texte unifiée (ticks + titre)
                    "color": self._font_color,
                }
                sb_defaults.update(sb_args)
                add_kwargs["scalar_bar_args"] = sb_defaults
            for opt in ("point_size", "line_width", "ambient", "specular", "diffuse"):
                if opt in prm:
                    add_kwargs[opt] = prm[opt]
            self._plotter.add_mesh(mesh, **{k: v for k, v in add_kwargs.items() if v is not None})
            if add_kwargs.get("show_scalar_bar"):
                self._scalar_bar_added = True

        # Ajuster caméra globale
        if self._datasets:
            self._plotter.camera_position = "xy"  # orientation initiale simple

    # ------------------------------
    # Rotations caméra robustes
    # ------------------------------
    def _rotate_camera(self, azimuth_deg: float = 0.0, elevation_deg: float = 0.0):
        """
        Applique une rotation caméra en essayant d'abord les méthodes VTK,
        sinon fallback par rotation de la position autour du focal point.
        """
        cam = self._plotter.camera
        # Azimuth
        if azimuth_deg:
            if callable(getattr(cam, "Azimuth", None)):
                cam.Azimuth(azimuth_deg)
            elif callable(getattr(cam, "azimuth", None)):
                cam.azimuth(azimuth_deg)
            else:
                self._rotate_camera_fallback(axis='z', angle_deg=azimuth_deg)
        # Elevation
        if elevation_deg:
            if callable(getattr(cam, "Elevation", None)):
                cam.Elevation(elevation_deg)
            elif callable(getattr(cam, "elevation", None)):
                cam.elevation(elevation_deg)
            else:
                self._rotate_camera_fallback(axis='x', angle_deg=elevation_deg)

    def _rotate_camera_fallback(self, axis: str, angle_deg: float):
        """Fallback: rotation de la position caméra autour du focal point."""
        cam = self._plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        vec = pos - fp
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        if axis == 'z':
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
        elif axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]])
        elif axis == 'y':
            R = np.array([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]])
        else:
            return
        new_pos = fp + R.dot(vec)
        cam.position = new_pos.tolist()
        # Optionnel: pas de modification du view_up ici (préserver stabilité)

    def _apply_rotation(self, azimuth_deg: float, elevation_deg: float):
        """Rotation + rendu (indispensable off_screen pour que la caméra se propage aux captures)."""
        self._rotate_camera(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        self._plotter.render()

    # ------------------------------
    # Affichage interactif
    # ------------------------------
    def show(self, **show_kwargs):
        self._ensure_plotter()
        return self._plotter.show(**show_kwargs)

    # ------------------------------
    # Capture d'écran
    # ------------------------------
    def capture_screenshot(self, output_path: Union[str, Path], transparent: bool = False):
        self._ensure_plotter()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = self._plotter.screenshot(filename=str(output_path), transparent_background=transparent)
        return img

    # ------------------------------
    # Enregistrement vidéo / GIF
    # ------------------------------
    @staticmethod
    def _ffmpeg_encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
        """Retourne True si l'encodeur ffmpeg est disponible."""
        try:
            out = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode != 0:
                return False
            # Chaque ligne des encoders commence par 2 lettres (flags) + espace + nom
            token = f" {encoder}"
            return any(line.strip().endswith(encoder) or f" {encoder} " in line for line in out.stdout.splitlines())
        except Exception:
            return False

    @staticmethod
    def _select_ffmpeg_codec(ffmpeg_bin: str, requested: str) -> Tuple[str, Optional[str]]:
        """
        (CONSERVÉ pour compat) -> retourne premier codec dispo ou fallback.
        Désormais remplacé par _codec_candidates, gardé si appelé ailleurs.
        """
        order = [
            requested,
            "libx264", "h264_nvenc", "h264_qsv", "h264_v4l2m2m", "h264_vaapi",
            "libx265", "hevc_nvenc", "hevc_qsv", "hevc_v4l2m2m", "hevc_vaapi",
            "libvpx_vp9", "libvpx_vp8",
            "mpeg4",
            "ffv1",   # sans perte si dispo
        ]
        seen = set()
        for c in order:
            if c in seen:
                continue
            seen.add(c)
            if VTKVisualizer._ffmpeg_encoder_available(ffmpeg_bin, c):
                if c == requested:
                    return c, None
                return c, f"Codec '{requested}' indisponible -> fallback '{c}'."
        # Aucun codec de la liste : laisser ffmpeg choisir (pas de -c:v)
        return "", f"Codec '{requested}' indisponible, aucun fallback trouvé -> utilisation du codec par défaut ffmpeg."

    @staticmethod
    def _codec_candidates(ffmpeg_bin: str, requested: str, prefer_software: bool = True) -> List[str]:
        """
        Retourne une liste ordonnée de codecs à essayer.
        - Si prefer_software=True, ne met les codecs hardware (nvenc, qsv, vaapi, v4l2m2m) qu’en fin.
        - Le codec explicitement demandé reste en tête (même si HW).
        """
        hw_suffix = ("_nvenc", "_qsv", "_vaapi", "_v4l2m2m")
        # Base logiciels (ordre de préférence)
        software_set = [
            "libx264", "libopenh264", "mpeg4", "libxvid",
            "libvpx_vp9", "libvpx_vp8", "ffv1", "rawvideo"
        ]
        # Encoders hardware potentiels
        hardware_set = [
            "h264_nvenc", "hevc_nvenc",
            "h264_qsv", "hevc_qsv",
            "h264_vaapi", "hevc_vaapi",
            "h264_v4l2m2m", "hevc_v4l2m2m",
        ]
        ordered = [requested] if requested else []
        if prefer_software:
            ordered += [c for c in software_set if c not in ordered]
            ordered += [c for c in hardware_set if c not in ordered]
        else:
            # Mélange requested -> hw -> software
            ordered += [c for c in hardware_set if c not in ordered]
            ordered += [c for c in software_set if c not in ordered]
        # Filtrer doublons
        uniq = []
        for c in ordered:
            if c and c not in uniq:
                uniq.append(c)
        # Garder uniquement ceux disponibles
        available = [c for c in uniq if VTKVisualizer._ffmpeg_encoder_available(ffmpeg_bin, c)]
        # Toujours ajouter fallback sans spécifier codec (marqueur "")
        available.append("")  # "" => laisser ffmpeg décider
        return available

    def record_rotation(
        self,
        output_path: Union[str, Path],
        n_frames: int = 180,
        step: float = 2.0,
        elevation: float = 0.0,
        gif: bool = False,
        fps: int = 30,
        quality: int = 5,
        window_size: Optional[Tuple[int, int]] = None,
        supersample: int = 1,
        anti_aliasing: bool = True,
        codec: str = "libx264",
        bitrate: Optional[str] = None,
        crf: Optional[int] = None,
        auto_codec_fallback: bool = True,
        prefer_software: bool = True,
    ):
        """
        Effectue une rotation azimutale de la caméra et enregistre.
        gif=True => GIF via imageio.
        gif=False => MP4/AVI via:
          1) imageio-ffmpeg si dispo (flux direct)
          2) sinon binaire ffmpeg si présent
          3) sinon erreur explicite

        Paramètres qualité supplémentaires:
          window_size: (w,h) taille finale souhaitée (sans supersample). Si None, taille actuelle.
          supersample: facteur (>1) pour rendre en (w*factor, h*factor) puis encoder (améliore netteté).
          anti_aliasing: active l'anti-aliasing (si supporté).
          codec: encodeur vidéo ffmpeg/imageio (ex: libx264, mpeg4, libvpx_vp9).
          bitrate: ex '8M' (si défini, ffmpeg utilisera ce débit).
          crf: qualité cible (0-51, plus bas = meilleur) si codec type x264/x265 (ignoré si bitrate défini).
          quality: param interne imageio (0-10) — conserver élevé (8-10) pour limiter la perte.
         auto_codec_fallback: si True, sélection automatique d'un codec disponible si le demandé manque.
        """
        self._ensure_plotter()
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".gif" if gif else ".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Gestion résolution / supersampling
        original_size = getattr(self._plotter, "window_size", None)
        target_size = None
        if window_size:
            w, h = window_size
            w = int(w)
            h = int(h)
            if supersample > 1:
                target_size = (w * supersample, h * supersample)
            else:
                target_size = (w, h)
            try:
                self._plotter.window_size = target_size
            except Exception:
                target_size = None  # fallback silencieux

        if anti_aliasing:
            with contextlib.suppress(Exception):
                self._plotter.enable_anti_aliasing()

        if gif and imageio is None:
            gif = False  # fallback (mais nécessitera quand même imageio pour vidéo)

        if gif:
            frames = []
            for i in range(n_frames):
                elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                frames.append(self._plotter.screenshot(return_img=True))
            imageio.mimsave(output_path, frames, fps=fps)
            # Restauration taille
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # --- Branche vidéo ---
        if imageio is None:
            raise RuntimeError("imageio indisponible. Installez: pip install imageio imageio-ffmpeg ou ffmpeg système.")

        # 1) Essai imageio-ffmpeg
        has_imageio_ffmpeg = False
        try:
            import imageio_ffmpeg  # noqa: F401
            has_imageio_ffmpeg = True
        except Exception:
            pass

        if has_imageio_ffmpeg:
            # Encodage en flux direct
            writer = None
            try:
                writer_kwargs = dict(fps=fps, quality=quality, codec=codec)
                if bitrate:
                    writer_kwargs["bitrate"] = bitrate
                out_params = []
                if crf is not None and not bitrate:
                    out_params += ["-crf", str(crf)]
                if out_params:
                    writer_kwargs["output_params"] = out_params
                writer = imageio.get_writer(str(output_path), fps=fps, quality=quality)
                # Remplacer par writer avec kwargs (compat anciennes versions)
                with contextlib.suppress(TypeError):
                    writer.close()
                    writer = imageio.get_writer(str(output_path), **writer_kwargs)
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    writer.append_data(frame)
            finally:
                if writer is not None:
                    with contextlib.suppress(Exception):
                        writer.close()
                self._plotter.close()
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 2) Fallback binaire ffmpeg (frames temporaires PNG)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            temp_dir = Path(tempfile.mkdtemp(prefix="vtkvis_frames_"))
            try:
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    imageio.imwrite(temp_dir / f"frame{i:05d}.png", frame)
                # Essais multi-codecs
                if auto_codec_fallback:
                    candidates = self._codec_candidates(ffmpeg_bin, codec, prefer_software=prefer_software)
                else:
                    candidates = [codec]
                last_err = None
                for idx, cdc in enumerate(candidates):
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-framerate", str(fps),
                        "-i", str(temp_dir / "frame%05d.png"),
                    ]
                    if cdc:
                        if idx == 0 and cdc != codec:
                            warnings.warn(f"Codec '{codec}' indisponible -> fallback '{cdc}'.")
                        elif idx > 0:
                            warnings.warn(f"Tentative codec fallback '{cdc}'.")
                        cmd += ["-c:v", cdc]
                    if crf is not None and not bitrate and (cdc.startswith("libx264") or cdc.startswith("libx265") or cdc in ("libvpx_vp9", "")):
                        cmd += ["-crf", str(crf)]
                    if bitrate:
                        cmd += ["-b:v", bitrate]
                    if cdc.startswith("libx264") or cdc.startswith("libx265"):
                        cmd += ["-preset", "slow"]
                    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        last_err = None
                        break
                    last_err = f"Codec '{cdc or 'auto'}' échec:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                if last_err:
                    raise RuntimeError("ffmpeg a échoué après tous les fallbacks:\n" + last_err)
            finally:
                self._plotter.close()
                with contextlib.suppress(Exception):
                    shutil.rmtree(temp_dir)
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 3) Erreur explicite
        raise RuntimeError(
            "Aucun backend vidéo fonctionnel.\n"
            "- Installez imageio-ffmpeg: pip install imageio-ffmpeg\n"
            "ou\n"
            "- Installez ffmpeg (ex: apt-get install ffmpeg)\n"
            "Sinon utilisez gif=True pour générer un GIF."
        )

    # ------------------------------
    # Utilitaires
    # ------------------------------
    def list_arrays(self) -> Dict[str, List[str]]:
        out = {}
        for ds, prm in self._datasets:
            out[prm.get("name")] = list(ds.array_names)
        return out

    @staticmethod
    def _choose_font_color(bg) -> str:
        """
        Retourne 'black' ou 'white' selon la luminance du background.
        Implémentation robuste sans dépendre de pyvista.parse_color (absent sur certaines versions).
        Accepte:
          - noms de couleurs (si matplotlib installé)
          - code hex (#RRGGBB ou #RGB)
          - tuple/list (r,g,b[,a]) avec composantes 0-1 ou 0-255
        """
        def _norm_rgb(rgb):
            if max(rgb) > 1.0:
                return [c / 255.0 for c in rgb]
            return rgb

        # Tentative matplotlib
        try:
            import matplotlib.colors as mcolors
            try:
                r, g, b = mcolors.to_rgb(bg)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                pass
        except Exception:
            pass

        # Tuples / listes
        if isinstance(bg, (tuple, list)) and 3 <= len(bg) <= 4:
            try:
                r, g, b = _norm_rgb(list(bg)[:3])
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                return "black"

        # Hex manuel
        if isinstance(bg, str) and bg.startswith("#"):
            h = bg.lstrip("#")
            if len(h) == 3:
                h = "".join([c*2 for c in h])
            if len(h) == 6:
                try:
                    r = int(h[0:2], 16) / 255.0
                    g = int(h[2:4], 16) / 255.0
                    b = int(h[4:6], 16) / 255.0
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    return "black" if lum > 0.6 else "white"
                except Exception:
                    return "black"

        # Fallback par défaut
        return "black"


# -------------------------------------------------
# Exemple d'utilisation (protégé par __main__)
# -------------------------------------------------
if __name__ == "__main__":
    vis = VTKVisualizer(background="black", off_screen=True)
    vis.add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-associations_model-MCM_space-HCP_tracto.vtk",
        {
            "display_array": "point_index",
            "cmap": "viridis",
            "threshold": ("point_index", (0, 24)),
            "opacity": 0.9,
            "scalar_bar": True,  # mappé vers show_scalar_bar
            "name": "associations",
        }
    ).add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-centroids_model-MCM_space-subject_tracto.vtk",
        {
            "display_array": None,
            "color": "red",
            "point_size": 20,
            "opacity": 1.0,
            "name": "centroids",
            "style": "points",
            "render_points_as_spheres": True,
        }
    )
    # vis.capture_screenshot("capture.png")
    vis.record_rotation("rotation.mp4", n_frames=240, step=1.5, quality=10)
    # vis.show()  # si rendu interactif possible
    # vis.show()  # si rendu interactif possible
