# Guide de Contribution - actiDep

Merci de votre intérêt pour contribuer au projet actiDep ! Ce guide vous aidera à démarrer.

## 📋 Table des Matières

- [Code de Conduite](#code-de-conduite)
- [Comment Contribuer](#comment-contribuer)
- [Standards de Développement](#standards-de-développement)
- [Processus de Soumission](#processus-de-soumission)
- [Structure du Projet](#structure-du-projet)
- [Tests](#tests)
- [Documentation](#documentation)

## 🤝 Code de Conduite

En participant à ce projet, vous acceptez de respecter notre code de conduite. Soyez respectueux et professionnel dans toutes vos interactions.

## 🛠️ Comment Contribuer

### Types de Contributions

Nous accueillons plusieurs types de contributions :

- **Correction de bugs** : Signaler et corriger des problèmes
- **Nouvelles fonctionnalités** : Proposer et implémenter de nouvelles fonctionnalités
- **Documentation** : Améliorer la documentation existante
- **Tests** : Ajouter ou améliorer les tests
- **Optimisations** : Améliorer les performances

### Signaler des Problèmes

Avant de signaler un problème :

1. Vérifiez que le problème n'existe pas déjà dans les [issues](https://github.com/nathandecaux/ActiDep/issues)
2. Utilisez la dernière version du code
3. Fournissez des informations détaillées sur votre environnement

**Template pour signaler un bug :**

```markdown
## Description du problème
[Description claire et concise du problème]

## Étapes pour reproduire
1. [Première étape]
2. [Deuxième étape]
3. [Voir l'erreur]

## Comportement attendu
[Description du comportement attendu]

## Environnement
- OS : [ex. Ubuntu 20.04]
- Python : [ex. 3.8.5]
- Version actiDep : [ex. 0.1.0]
- Dépendances : [FSL, MRtrix, etc.]

## Informations supplémentaires
[Logs, captures d'écran, etc.]
```

## 📚 Standards de Développement

### Style de Code

- **PEP 8** : Suivre les conventions de style Python
- **Docstrings** : Utiliser le format Google/NumPy pour la documentation
- **Type Hints** : Utiliser les annotations de type quand approprié
- **Noms** : Utiliser des noms descriptifs pour les variables et fonctions

**Exemple de docstring :**

```python
def evaluate_along_streamlines(scalar_img, streamlines, nr_points, **kwargs):
    """
    Évalue une métrique scalaire le long des streamlines d'un faisceau.
    
    Parameters
    ----------
    scalar_img : numpy.ndarray
        Image 3D contenant les valeurs scalaires à évaluer
    streamlines : list or dipy.tracking.streamline.Streamlines
        Liste des streamlines du faisceau
    nr_points : int
        Nombre de points pour l'échantillonnage le long des streamlines
    **kwargs : dict
        Arguments supplémentaires (algorithm, beginnings, etc.)
    
    Returns
    -------
    tuple[list, list]
        Tuple contenant les moyennes et écarts-types le long du faisceau
        
    Examples
    --------
    >>> means, stds = evaluate_along_streamlines(fa_img, streamlines, 100)
    >>> print(f"Nombre de points: {len(means)}")
    """
```

### Structure des Modules

Chaque module doit suivre cette structure :

```python
"""
Description du module.

Ce module contient des fonctions pour...
"""

# Imports standard library
import os
import sys

# Imports third-party
import numpy as np
import nibabel as nib

# Imports locaux
from actiDep.utils.tools import run_cli_command
from actiDep.data.loader import ActiDepFile

# Constants
DEFAULT_THRESHOLD = 100.0

# Functions
def main_function():
    """Function principale du module."""
    pass

# Main execution
if __name__ == "__main__":
    main_function()
```

### Gestion des Erreurs

- Utiliser des exceptions spécifiques
- Fournir des messages d'erreur clairs
- Logger les erreurs appropriées

```python
import logging

logger = logging.getLogger(__name__)

def process_file(file_path):
    """Traite un fichier avec gestion d'erreur appropriée."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    
    try:
        # Traitement du fichier
        result = complex_processing(file_path)
        return result
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {file_path}: {e}")
        raise
```

## 🔄 Processus de Soumission

### 1. Fork et Clone

```bash
# Fork le repo sur GitHub, puis :
git clone https://github.com/VOTRE-USERNAME/ActiDep.git
cd ActiDep
```

### 2. Créer une Branche

```bash
# Créer une branche descriptive
git checkout -b feature/nouvelle-fonctionnalite
# ou
git checkout -b fix/correction-bug
```

### 3. Développement

```bash
# Installation en mode développement
pip install -e .

# Faire vos modifications
# Tester localement
python -m pytest tests/

# Vérifier le style
flake8 actiDep/
black actiDep/
```

### 4. Commit et Push

```bash
# Commits atomiques avec messages clairs
git add .
git commit -m "feat: ajouter fonction de clustering automatique

- Implémentation de QuickBundles avec paramètres adaptatifs
- Tests unitaires pour la nouvelle fonctionnalité
- Documentation mise à jour"

git push origin feature/nouvelle-fonctionnalite
```

### 5. Pull Request

- Créer une Pull Request sur GitHub
- Utiliser le template fourni
- Lier aux issues pertinentes
- Attendre la review

**Template Pull Request :**

```markdown
## Description
[Description claire des changements]

## Type de changement
- [ ] Bug fix
- [ ] Nouvelle fonctionnalité
- [ ] Breaking change
- [ ] Documentation

## Tests
- [ ] Tests existants passent
- [ ] Nouveaux tests ajoutés
- [ ] Tests manuels effectués

## Checklist
- [ ] Code suit PEP 8
- [ ] Docstrings ajoutées/mises à jour
- [ ] Documentation mise à jour
- [ ] Changelog mis à jour (si applicable)
```

## 🧪 Tests

### Structure des Tests

```
tests/
├── __init__.py
├── conftest.py              # Configuration pytest
├── test_core.py             # Tests du module core
├── test_data/               # Données de test
│   ├── sample.nii.gz
│   └── sample.trk
├── analysis/
│   ├── test_tractometry.py
│   └── test_microstructure.py
├── utils/
│   ├── test_tools.py
│   └── test_tractography.py
└── integration/
    └── test_full_pipeline.py
```

### Écriture des Tests

```python
import pytest
import numpy as np
from actiDep.analysis.tractometry import evaluate_along_streamlines

class TestTractometry:
    """Tests pour le module tractometry."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        self.scalar_img = np.random.rand(64, 64, 64)
        self.streamlines = [np.random.rand(100, 3) for _ in range(10)]
    
    def test_evaluate_along_streamlines_basic(self):
        """Test basique de evaluate_along_streamlines."""
        means, stds = evaluate_along_streamlines(
            self.scalar_img, 
            self.streamlines, 
            nr_points=50
        )
        
        assert len(means) == 50
        assert len(stds) == 50
        assert all(isinstance(m, (int, float)) for m in means)
    
    def test_evaluate_along_streamlines_empty(self):
        """Test avec des streamlines vides."""
        with pytest.raises(ValueError):
            evaluate_along_streamlines(self.scalar_img, [], nr_points=50)
    
    @pytest.mark.parametrize("algorithm", ["equal_dist", "distance_map", "afq"])
    def test_different_algorithms(self, algorithm):
        """Test des différents algorithmes."""
        means, stds = evaluate_along_streamlines(
            self.scalar_img,
            self.streamlines,
            nr_points=50,
            algorithm=algorithm
        )
        assert len(means) == 50
```

### Exécution des Tests

```bash
# Tous les tests
pytest

# Tests spécifiques
pytest tests/analysis/test_tractometry.py

# Tests avec couverture
pytest --cov=actiDep tests/

# Tests avec rapport détaillé
pytest -v --tb=short
```

## 📖 Documentation

### Types de Documentation

1. **Docstrings** : Documentation des fonctions et classes
2. **README** : Documentation utilisateur principale
3. **Guides** : Tutoriels et guides d'utilisation
4. **API Reference** : Documentation technique complète

### Format des Docstrings

Utiliser le format NumPy/SciPy :

```python
def complex_function(param1, param2, option=None):
    """
    Fonction complexe qui fait quelque chose d'important.
    
    Cette fonction combine param1 et param2 pour produire un résultat
    selon l'option spécifiée.
    
    Parameters
    ----------
    param1 : str
        Premier paramètre de type string
    param2 : int
        Deuxième paramètre de type entier
    option : {'method1', 'method2'}, optional
        Méthode à utiliser pour le calcul (default: 'method1')
    
    Returns
    -------
    result : dict
        Dictionnaire contenant :
        - 'value' : float, valeur calculée
        - 'status' : str, statut du calcul
    
    Raises
    ------
    ValueError
        Si param2 est négatif
    TypeError
        Si param1 n'est pas une string
    
    See Also
    --------
    related_function : Fonction connexe
    
    Notes
    -----
    Cette fonction utilise l'algorithme XYZ décrit dans [1]_.
    
    References
    ----------
    .. [1] Smith, J. "Algorithm XYZ", Journal of Algorithms, 2020.
    
    Examples
    --------
    >>> result = complex_function("test", 42)
    >>> print(result['value'])
    3.14159
    """
```

### Génération de Documentation

La documentation peut être générée avec Sphinx :

```bash
# Installation de Sphinx
pip install sphinx sphinx-rtd-theme

# Génération de la documentation
cd docs/
make html
```

## 🚀 Workflow de Développement

### Branches

- `main` : Branche principale stable
- `develop` : Branche de développement
- `feature/*` : Nouvelles fonctionnalités
- `fix/*` : Corrections de bugs
- `docs/*` : Mises à jour documentation

### Git Hooks

Configurez des hooks pre-commit pour automatiser les vérifications :

```bash
# Installation de pre-commit
pip install pre-commit

# Configuration (.pre-commit-config.yaml)
repos:
- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
  - id: black
- repo: https://github.com/pycqa/flake8
  rev: 4.0.1
  hooks:
  - id: flake8
```

### Versioning

Nous suivons le [Semantic Versioning](https://semver.org/) :

- `MAJOR.MINOR.PATCH`
- Incrémentation MAJOR pour les breaking changes
- Incrémentation MINOR pour les nouvelles fonctionnalités
- Incrémentation PATCH pour les corrections de bugs

## 📞 Support

Si vous avez des questions :

1. Consultez la [documentation](README.md)
2. Cherchez dans les [issues existantes](https://github.com/nathandecaux/ActiDep/issues)
3. Créez une nouvelle issue avec le label `question`
4. Contactez l'équipe : [nathan.decaux@irisa.fr](mailto:nathan.decaux@irisa.fr)

## 🙏 Remerciements

Merci à tous les contributeurs qui rendent ce projet possible !

- Testeurs et utilisateurs qui signalent des bugs
- Développeurs qui soumettent des améliorations
- Équipe EMPENN pour le support scientifique
