# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation complète du projet
- Guide de contribution (CONTRIBUTING.md)
- Structure modulaire pour l'analyse de diffusion

### Changed
- Mise à jour du README avec documentation détaillée

### Fixed
- Correction des liens de navigation dans la documentation

## [0.1.0] - 2024-07-11

### Added
- Version initiale du package actiDep
- Module `analysis` pour tractométrie et analyse microstructurelle
  - `tractometry.py` : Analyse tractométrique avec multiple algorithmes
  - `fiber_density.py` : Analyse de densité de fibres
  - `microstructure.py` : Analyse des paramètres microstructurels
- Module `data` pour gestion des données BIDS
  - `loader.py` : Système de chargement compatible BIDS
  - `io.py` : Fonctions d'entrées/sorties
- Module `pipeline` pour les pipelines de traitement
  - `msmt_csd.py` : Pipeline Multi-Shell Multi-Tissue CSD
  - `bundle_seg.py` : Segmentation automatique de faisceaux
  - `preprocessing.py` : Prétraitement des données DWI
- Module `utils` pour utilitaires
  - `tractography.py` : Génération et manipulation de tractogrammes
  - `registration.py` : Recalage d'images
  - `mcm.py` : Utilitaires pour modèles MCM
  - `fod.py` : Fonctions d'orientation de fibres
- Module `visualisation` pour affichage
  - `centroids_params.py` : Interface interactive clustering
  - `atlas_visu.py` : Visualisation d'atlas
- Module `atlasing` pour création d'atlas
  - `HCP_atlasing.py` : Atlasing basé sur HCP
  - `HCP_averaging.py` : Moyennage de tractogrammes
- Interface utilisateur basique
  - `browser.py` : Navigateur de données
  - `tractometry_dashboard.py` : Dashboard tractométrie
- Configuration automatique des outils externes
  - Support ANIMA, MRtrix, TractSeg, ANTs
- Tests unitaires pour modules principaux
- Support formats : NIfTI, TRK, TCK, VTK

### Dependencies
- numpy : Calculs numériques
- pandas : Manipulation de données
- nibabel : Lecture/écriture images neuroimagerie
- dipy : Traitement diffusion
- scipy : Calculs scientifiques
- matplotlib : Visualisation
- vtk : Manipulation données 3D
- ants : Recalage d'images
- SimpleITK : Traitement d'images

### Infrastructure
- Configuration setuptools et Poetry
- Structure package compatible pip
- Support Python 3.8+
- Intégration BIDS (Brain Imaging Data Structure)

## Types de changements

- `Added` : Nouvelles fonctionnalités
- `Changed` : Modifications de fonctionnalités existantes  
- `Deprecated` : Fonctionnalités qui seront supprimées
- `Removed` : Fonctionnalités supprimées
- `Fixed` : Corrections de bugs
- `Security` : Corrections de sécurité
