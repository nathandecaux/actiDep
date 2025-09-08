# API Reference - actiDep

Documentation technique complète de l'API du package `actiDep`.

## Modules Principaux

### actiDep.data.loader

#### Classe `ActiDepFile`

Représente un fichier dans un dataset BIDS avec fonctionnalités étendues.

```python
class ActiDepFile:
    """
    Classe pour représenter un fichier BIDS avec métadonnées.
    
    Attributes
    ----------
    path : str
        Chemin absolu vers le fichier
    filename : str
        Nom du fichier
    scope : str
        'raw' ou 'derivatives'
    pipeline : str or None
        Nom du pipeline de traitement
    datatype : str
        Type de données (dwi, anat, etc.)
    entities : dict
        Dictionnaire des entités BIDS
    """
    
    def __init__(self, path: str, subject: Optional['Subject'] = None):
        """
        Initialise un objet ActiDepFile.
        
        Parameters
        ----------
        path : str
            Chemin vers le fichier
        subject : Subject, optional
            Objet Subject parent
        """
    
    def get_entities(self, **kwargs) -> dict:
        """
        Retourne les entités de base du fichier.
        
        Returns
        -------
        dict
            Dictionnaire des entités
        """
    
    def get_full_entities(self, **kwargs) -> dict:
        """
        Retourne toutes les entités incluant scope, pipeline, datatype.
        
        Returns
        -------
        dict
            Dictionnaire complet des entités
        """
    
    def copy(self, dest: str) -> str:
        """
        Copie le fichier vers une destination.
        
        Parameters
        ----------
        dest : str
            Chemin de destination
            
        Returns
        -------
        str
            Chemin du fichier copié
        """
    
    def move(self, dest: str) -> str:
        """
        Déplace le fichier vers une destination.
        
        Parameters
        ----------
        dest : str
            Chemin de destination
            
        Returns
        -------
        str
            Chemin du fichier déplacé
        """
```

#### Classe `Subject`

Interface pour manipuler les données d'un sujet individuel.

```python
class Subject:
    """
    Représente un sujet dans un dataset BIDS.
    
    Attributes
    ----------
    sub_id : str
        Identifiant du sujet
    db_root : str
        Chemin racine du dataset
    """
    
    def __init__(self, sub_id: str, db_root: Optional[str] = None):
        """
        Initialise un objet Subject.
        
        Parameters
        ----------
        sub_id : str
            Identifiant du sujet (sans préfixe 'sub-')
        db_root : str, optional
            Chemin vers le dataset BIDS
        """
    
    def get(self, **kwargs) -> List[ActiDepFile]:
        """
        Récupère des fichiers selon des critères.
        
        Parameters
        ----------
        **kwargs
            Critères de filtrage (pipeline, suffix, metric, etc.)
            
        Returns
        -------
        List[ActiDepFile]
            Liste des fichiers correspondants
            
        Examples
        --------
        >>> fa_files = subject.get(pipeline='anima_preproc', metric='FA')
        >>> tractos = subject.get(pipeline='bundle_seg', suffix='tracto')
        """
    
    def get_unique(self, **kwargs) -> ActiDepFile:
        """
        Récupère un fichier unique selon des critères.
        
        Parameters
        ----------
        **kwargs
            Critères de filtrage
            
        Returns
        -------
        ActiDepFile
            Fichier unique correspondant
            
        Raises
        ------
        ValueError
            Si aucun fichier ou plusieurs fichiers trouvés
        """
    
    def write_object(self, obj, **entities):
        """
        Écrit un objet dans le système de fichiers.
        
        Parameters
        ----------
        obj
            Objet à sauvegarder
        **entities
            Entités pour construire le chemin
        """
```

#### Classe `Actidep`

Interface pour manipuler un dataset BIDS complet.

```python
class Actidep:
    """
    Interface pour un dataset BIDS complet.
    
    Attributes
    ----------
    root : str
        Chemin racine du dataset
    """
    
    def __init__(self, root: str):
        """
        Initialise un dataset.
        
        Parameters
        ----------
        root : str
            Chemin vers le dataset BIDS
        """
    
    def get_subjects(self) -> List[str]:
        """
        Retourne la liste des sujets.
        
        Returns
        -------
        List[str]
            Liste des identifiants de sujets
        """
    
    def get_pipelines(self) -> List[str]:
        """
        Retourne la liste des pipelines disponibles.
        
        Returns
        -------
        List[str]
            Liste des noms de pipelines
        """
    
    def get(self, subject: str, **kwargs) -> List[ActiDepFile]:
        """
        Récupère des fichiers pour un sujet.
        
        Parameters
        ----------
        subject : str
            Identifiant du sujet
        **kwargs
            Critères de filtrage
            
        Returns
        -------
        List[ActiDepFile]
            Liste des fichiers
        """
    
    def get_global(self, **kwargs) -> List[ActiDepFile]:
        """
        Récupère des fichiers pour tous les sujets.
        
        Parameters
        ----------
        **kwargs
            Critères de filtrage
            
        Returns
        -------
        List[ActiDepFile]
            Liste des fichiers pour tous les sujets
        """
```

### actiDep.analysis.tractometry

Module principal pour l'analyse tractométrique.

#### `evaluate_along_streamlines`

```python
def evaluate_along_streamlines(
    scalar_img: np.ndarray,
    streamlines: Union[List, Streamlines],
    nr_points: int,
    beginnings: Optional[np.ndarray] = None,
    dilate: int = 0,
    predicted_peaks: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    algorithm: str = "distance_map"
) -> Tuple[List[float], List[float]]:
    """
    Évalue une métrique scalaire le long des streamlines.
    
    Parameters
    ----------
    scalar_img : np.ndarray
        Image 3D contenant les valeurs scalaires
    streamlines : List or Streamlines
        Streamlines à analyser
    nr_points : int
        Nombre de points pour l'échantillonnage
    beginnings : np.ndarray, optional
        Masque des débuts de faisceaux
    dilate : int, default=0
        Nombre d'itérations de dilatation
    predicted_peaks : np.ndarray, optional
        Pics prédits pour orientation
    affine : np.ndarray, optional
        Matrice affine de l'image
    algorithm : str, default="distance_map"
        Algorithme d'agrégation:
        - "equal_dist": échantillonnage équidistant
        - "distance_map": agrégation par cKDTree
        - "cutting_plane": plans de coupe
        - "afq": méthode AFQ
    
    Returns
    -------
    Tuple[List[float], List[float]]
        (moyennes, écarts-types) le long du faisceau
    
    Examples
    --------
    >>> means, stds = evaluate_along_streamlines(
    ...     fa_data, streamlines, 100, algorithm="distance_map"
    ... )
    """
```

#### `process_projection`

```python
def process_projection(
    tracto_dict: Dict[str, ActiDepFile],
    metric_dict: Dict[str, ActiDepFile],
    beginnings_dict: Optional[Dict[str, ActiDepFile]] = None,
    **kwargs
) -> Dict[str, ActiDepFile]:
    """
    Traite la projection de métriques sur des faisceaux.
    
    Parameters
    ----------
    tracto_dict : Dict[str, ActiDepFile]
        Dictionnaire {nom_faisceau: fichier_tractogramme}
    metric_dict : Dict[str, ActiDepFile]
        Dictionnaire {nom_faisceau: fichier_métrique}
    beginnings_dict : Dict[str, ActiDepFile], optional
        Dictionnaire des débuts de faisceaux
    **kwargs
        Options supplémentaires (nr_points, algorithm, etc.)
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Dictionnaire des fichiers de résultats
    
    Examples
    --------
    >>> results = process_projection(
    ...     tracto_dict={'CST': cst_file},
    ...     metric_dict={'CST': fa_file},
    ...     nr_points=100
    ... )
    """
```

#### `process_tractseg_analysis`

```python
def process_tractseg_analysis(
    subjects_txt: str,
    dataset_path: str = "/path/to/bids",
    with_3dplot: bool = False,
    metric: str = 'FA'
) -> None:
    """
    Lance une analyse tractométrique complète avec TractSeg.
    
    Parameters
    ----------
    subjects_txt : str
        Chemin vers le fichier liste des sujets
    dataset_path : str
        Chemin vers le dataset BIDS
    with_3dplot : bool, default=False
        Inclure des visualisations 3D
    metric : str, default='FA'
        Métrique à analyser ('FA', 'MD', 'RD', 'AD')
    
    Examples
    --------
    >>> process_tractseg_analysis(
    ...     "subjects.txt",
    ...     metric='FA',
    ...     with_3dplot=True
    ... )
    """
```

### actiDep.utils.tractography

Utilitaires pour la tractographie.

#### `generate_ifod2_tracto`

```python
def generate_ifod2_tracto(
    odf: ActiDepFile,
    seeds: ActiDepFile,
    **kwargs
) -> Dict[str, ActiDepFile]:
    """
    Génère une tractographie iFOD2 avec MRtrix.
    
    Parameters
    ----------
    odf : ActiDepFile
        Fichier des fonctions d'orientation de fibres
    seeds : ActiDepFile
        Fichier de masque de graines
    **kwargs
        Options pour tckgen
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Dictionnaire avec le tractogramme généré
    
    Examples
    --------
    >>> odf = subject.get_unique(suffix='fod', label='WM')
    >>> seeds = subject.get_unique(suffix='mask', label='brain')
    >>> tracto = generate_ifod2_tracto(odf, seeds)
    """
```

#### `get_tractogram_endings`

```python
def get_tractogram_endings(
    tractogram_file: Union[str, ActiDepFile],
    reference: Union[str, ActiDepFile]
) -> Dict[str, ActiDepFile]:
    """
    Calcule les extrémités des streamlines.
    
    Parameters
    ----------
    tractogram_file : str or ActiDepFile
        Fichier tractogramme
    reference : str or ActiDepFile
        Image de référence
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Dictionnaire avec masques de début et fin
    
    Examples
    --------
    >>> endings = get_tractogram_endings(tracto_file, fa_file)
    >>> start_mask = endings['start']
    >>> end_mask = endings['end']
    """
```

### actiDep.pipeline.msmt_csd

Pipeline Multi-Shell Multi-Tissue CSD.

#### `process_response`

```python
def process_response(
    subject: Subject,
    dwi_data: Dict[str, ActiDepFile],
    pipeline: str,
    **kwargs
) -> Dict[str, ActiDepFile]:
    """
    Calcule les réponses tissulaires pour MSMT-CSD.
    
    Parameters
    ----------
    subject : Subject
        Objet sujet
    dwi_data : Dict[str, ActiDepFile]
        Données DWI ('dwi', 'bval', 'bvec')
    pipeline : str
        Nom du pipeline
    **kwargs
        Options pour dwi2response
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Fichiers de réponses tissulaires
    
    Examples
    --------
    >>> responses = process_response(
    ...     subject, dwi_data, "msmt_csd"
    ... )
    """
```

#### `process_fod`

```python
def process_fod(
    subject: Subject,
    dwi_data: Dict[str, ActiDepFile],
    pipeline: str,
    **kwargs
) -> Dict[str, ActiDepFile]:
    """
    Calcule les FOD avec MSMT-CSD.
    
    Parameters
    ----------
    subject : Subject
        Objet sujet
    dwi_data : Dict[str, ActiDepFile]
        Données DWI
    pipeline : str
        Nom du pipeline
    **kwargs
        Options pour dwi2fod
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Fichiers FOD par tissu
    
    Examples
    --------
    >>> fods = process_fod(subject, dwi_data, "msmt_csd")
    >>> wm_fod = fods['WM']
    >>> gm_fod = fods['GM']
    """
```

### actiDep.utils.mcm

Utilitaires pour les modèles Multi-Compartiment.

#### `project_to_central_line_from_vtk`

```python
def project_to_central_line_from_vtk(
    vtk_file_path: str,
    reference_nifti_path: str,
    output_path: str,
    num_points_central_line: int = 100,
    transformation_matrix: Optional[np.ndarray] = None,
    transformation_center: Optional[np.ndarray] = None,
    rotate_z_180: bool = False,
    center_vtk: bool = True
) -> None:
    """
    Projette un tractogramme VTK sur une ligne centrale.
    
    Parameters
    ----------
    vtk_file_path : str
        Chemin vers le fichier VTK
    reference_nifti_path : str
        Image de référence NIfTI
    output_path : str
        Chemin de sortie
    num_points_central_line : int, default=100
        Nombre de points sur la ligne centrale
    transformation_matrix : np.ndarray, optional
        Matrice de transformation
    transformation_center : np.ndarray, optional
        Centre de transformation
    rotate_z_180 : bool, default=False
        Rotation de 180° en Z
    center_vtk : bool, default=True
        Centrer le VTK
    
    Examples
    --------
    >>> project_to_central_line_from_vtk(
    ...     "tracto.vtk", "reference.nii.gz", "output.vtk"
    ... )
    """
```

### actiDep.visualisation.centroids_params

Interface de clustering interactif.

#### Classe `TractoClusteringApp`

```python
class TractoClusteringApp:
    """
    Application interactive pour clustering de tractogrammes.
    
    Attributes
    ----------
    streamlines : Streamlines
        Streamlines chargées
    data : np.ndarray
        Données anatomiques
    affine : np.ndarray
        Matrice affine
    """
    
    def __init__(self):
        """Initialise l'application."""
    
    def load_data(
        self,
        tractogram_file: str,
        anatomy_file: Optional[str] = None
    ) -> None:
        """
        Charge un tractogramme et une image anatomique.
        
        Parameters
        ----------
        tractogram_file : str
            Chemin vers le tractogramme
        anatomy_file : str, optional
            Chemin vers l'image anatomique
        
        Examples
        --------
        >>> app = TractoClusteringApp()
        >>> app.load_data("tracto.trk", "fa.nii.gz")
        """
    
    def cluster_streamlines(
        self,
        threshold: float = 100.0
    ) -> QuickBundles:
        """
        Effectue le clustering avec QuickBundles.
        
        Parameters
        ----------
        threshold : float, default=100.0
            Seuil de distance pour clustering
        
        Returns
        -------
        QuickBundles
            Objet de clustering
        """
    
    def visualize_clusters(
        self,
        clusters: QuickBundles,
        show_centroids: bool = True
    ) -> None:
        """
        Visualise les résultats du clustering.
        
        Parameters
        ----------
        clusters : QuickBundles
            Résultats du clustering
        show_centroids : bool, default=True
            Afficher les centroïdes
        """
```

### actiDep.set_config

Configuration des outils externes.

#### `set_config`

```python
def set_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Configure les outils externes (ANIMA, TractSeg, MRtrix).
    
    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        (config, tools) - Dictionnaires de configuration
    
    Raises
    ------
    FileNotFoundError
        Si les fichiers de configuration n'existent pas
    
    Examples
    --------
    >>> config, tools = set_config()
    >>> anima_dir = config['animaDir']
    >>> mrtrix_bin = tools['mrtrix']
    """
```

#### `get_HCP_bundle_names`

```python
def get_HCP_bundle_names() -> Dict[str, str]:
    """
    Retourne le mapping des noms de faisceaux HCP.
    
    Returns
    -------
    Dict[str, str]
        Dictionnaire {nom_court: nom_complet}
    
    Examples
    --------
    >>> bundle_names = get_HCP_bundle_names()
    >>> print(bundle_names['CSTleft'])
    'Corticospinal tract left'
    """
```

### actiDep.utils.tools

Utilitaires génériques.

#### `run_cli_command`

```python
def run_cli_command(
    command: str,
    inputs: Dict[str, ActiDepFile],
    output_patterns: Dict[str, Dict],
    entities_template: Dict[str, Any],
    command_args: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, ActiDepFile]:
    """
    Execute une commande en ligne de commande.
    
    Parameters
    ----------
    command : str
        Nom de la commande
    inputs : Dict[str, ActiDepFile]
        Fichiers d'entrée
    output_patterns : Dict[str, Dict]
        Patterns des fichiers de sortie
    entities_template : Dict[str, Any]
        Template des entités
    command_args : List[str], optional
        Arguments de la commande
    **kwargs
        Options supplémentaires
    
    Returns
    -------
    Dict[str, ActiDepFile]
        Fichiers de sortie générés
    
    Examples
    --------
    >>> results = run_cli_command(
    ...     'dwi2fod',
    ...     inputs={'dwi': dwi_file},
    ...     output_patterns={'fod.nii.gz': {'suffix': 'fod'}},
    ...     entities_template=dwi_file.get_entities(),
    ...     command_args=['msmt_csd', '$dwi', 'fod.nii.gz']
    ... )
    """
```

## Types et Constantes

### Types Personnalisés

```python
from typing import Union, List, Dict, Optional, Tuple, Any
from pathlib import Path

# Types pour les fichiers
FilePath = Union[str, Path, ActiDepFile]
FileList = List[ActiDepFile]
FileDict = Dict[str, ActiDepFile]

# Types pour les données
ImageData = np.ndarray
StreamlineData = Union[List[np.ndarray], 'Streamlines']
MetricValues = List[float]

# Types pour les résultats
TractometryResult = Tuple[MetricValues, MetricValues]  # (means, stds)
ProcessingResult = Dict[str, ActiDepFile]
```

### Constantes

```python
# Algorithmes de tractométrie
TRACTOMETRY_ALGORITHMS = [
    "equal_dist",
    "distance_map", 
    "cutting_plane",
    "afq"
]

# Métriques supportées
SUPPORTED_METRICS = [
    "FA",  # Anisotropie fractionnelle
    "MD",  # Diffusivité moyenne
    "RD",  # Diffusivité radiale
    "AD",  # Diffusivité axiale
    "MK",  # Kurtosis moyenne
    "AK",  # Kurtosis axiale
    "RK"   # Kurtosis radiale
]

# Extensions de fichiers
TRACTOGRAM_EXTENSIONS = [".trk", ".tck", ".vtk", ".ply"]
IMAGE_EXTENSIONS = [".nii", ".nii.gz", ".nrrd"]
GRADIENT_EXTENSIONS = [".bval", ".bvec", ".b"]

# Faisceaux HCP
HCP_BUNDLES = [
    "AF_left", "AF_right", "ATR_left", "ATR_right",
    "CC", "CG_left", "CG_right", "CST_left", "CST_right",
    "FPT_left", "FPT_right", "FX_left", "FX_right",
    "ICP_left", "ICP_right", "IFO_left", "IFO_right",
    "ILF_left", "ILF_right", "MCP", "MLF_left", "MLF_right",
    "OR_left", "OR_right", "POPT_left", "POPT_right",
    "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right",
    "SLF_II_left", "SLF_II_right", "SLF_III_left", "SLF_III_right",
    "STR_left", "STR_right", "UF_left", "UF_right"
]
```

## Gestion des Erreurs

### Exceptions Personnalisées

```python
class ActiDepError(Exception):
    """Exception de base pour actiDep."""
    pass

class ConfigurationError(ActiDepError):
    """Erreur de configuration."""
    pass

class DataLoadingError(ActiDepError):
    """Erreur de chargement de données."""
    pass

class ProcessingError(ActiDepError):
    """Erreur de traitement."""
    pass

class ValidationError(ActiDepError):
    """Erreur de validation."""
    pass
```

### Codes de Retour

```python
# Codes de succès
SUCCESS = 0
PARTIAL_SUCCESS = 1

# Codes d'erreur
ERROR_CONFIGURATION = 10
ERROR_DATA_LOADING = 20
ERROR_PROCESSING = 30
ERROR_VALIDATION = 40
ERROR_IO = 50
```

## Configuration et Logging

### Configuration par Défaut

```python
DEFAULT_CONFIG = {
    'n_points_tractometry': 100,
    'tractometry_algorithm': 'distance_map',
    'clustering_threshold': 100.0,
    'n_cores': 4,
    'temp_dir': '/tmp',
    'output_format': 'csv'
}
```

### Logging

```python
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('actiDep')

# Usage dans le code
logger.info("Démarrage de l'analyse tractométrique")
logger.warning("Fichier manquant, utilisation des valeurs par défaut")
logger.error("Erreur lors du traitement du sujet")
```

Cette API reference fournit une documentation technique complète pour développer avec `actiDep`.
