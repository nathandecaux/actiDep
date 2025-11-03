# Configuration d'Exemple - actiDep

Ce dossier contient des exemples de configuration pour le package `actiDep`.

## Structure des Fichiers de Configuration

### ~/.anima/config.txt

Configuration pour les outils ANIMA :

```ini
[anima-scripts]
anima = /usr/local/bin/anima
extra-data-root = /usr/local/share/anima/data
anima-scripts-public-root = /usr/local/share/anima/scripts/public
anima-scripts-root = /usr/local/share/anima/scripts
```

### ~/.tractseg-config/config.txt

Configuration pour TractSeg et MRtrix :

```ini
[tractseg]
tractseg-bin = /usr/local/bin
mrtrix-bin = /usr/local/bin
```

## Variables d'Environnement

Ajoutez ces lignes à votre `~/.bashrc` ou `~/.zshrc` :

```bash
# actiDep Configuration
export ACTIDEP_DATA="/path/to/actidep/data"
export ACTIDEP_OUTPUT="/path/to/output"
export ACTIDEP_TEMP="/tmp/actidep"

# External Tools
export ANIMA_DIR="/usr/local/bin/anima"
export MRTRIX_DIR="/usr/local/bin"
export TRACTSEG_DIR="/usr/local/bin"
export ANTS_DIR="/usr/local/bin"

# FSL Configuration
export FSLDIR="/usr/local/fsl"
export PATH="${FSLDIR}/bin:${PATH}"
source ${FSLDIR}/etc/fslconf/fsl.sh

# Add to Python path
export PYTHONPATH="/path/to/actiDep:${PYTHONPATH}"
```

## Configuration Python

### config.py

Fichier de configuration Python pour paramètres avancés :

```python
"""
Configuration par défaut pour actiDep.
"""

import os
from pathlib import Path

# Chemins par défaut
DEFAULT_PATHS = {
    'data_root': os.environ.get('ACTIDEP_DATA', '/data/actidep'),
    'output_root': os.environ.get('ACTIDEP_OUTPUT', '/output/actidep'),
    'temp_root': os.environ.get('ACTIDEP_TEMP', '/tmp/actidep'),
    'cache_root': '~/.cache/actidep'
}

# Paramètres de traitement
PROCESSING_PARAMS = {
    'n_cores': int(os.environ.get('ACTIDEP_NCORES', '4')),
    'memory_limit': '8GB',
    'temp_cleanup': True,
    'verbose': True
}

# Paramètres tractométrie
TRACTOMETRY_PARAMS = {
    'default_algorithm': 'distance_map',
    'nr_points': 100,
    'clustering_threshold': 100.0,
    'dilate_endings': 0
}

# Paramètres visualization
VISUALIZATION_PARAMS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colormap': 'viridis',
    'save_format': 'png'
}

# Paramètres pipeline
PIPELINE_PARAMS = {
    'msmt_csd': {
        'lmax': 8,
        'algorithm': 'msmt_csd',
        'shell_selection': True
    },
    'bundle_seg': {
        'method': 'recobundles',
        'threshold': 8.0,
        'model': 'hcp_atlas'
    },
    'tractography': {
        'algorithm': 'ifod2',
        'n_streamlines': 1000000,
        'step_size': 0.5,
        'cutoff': 0.06
    }
}

# Formats de fichiers supportés
SUPPORTED_FORMATS = {
    'images': ['.nii', '.nii.gz', '.nrrd', '.mhd'],
    'tractograms': ['.trk', '.tck', '.vtk', '.ply'],
    'gradients': ['.bval', '.bvec', '.b'],
    'transforms': ['.mat', '.txt', '.h5'],
    'results': ['.csv', '.json', '.xlsx']
}

# Configuration logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'actidep.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
```

### subjects.txt

Format de fichier pour liste de sujets :

```text
/path/to/tracto/directory
bundle=AF_left AF_right ATR_left ATR_right CC CG_left CG_right CST_left CST_right
plot_3D=TEMPDIR/3d_plot
sub-01001 group1
sub-01002 group1  
sub-01003 group2
sub-01004 group2
sub-01005 group1
```

### dataset_description.json

Description du dataset BIDS :

```json
{
    "Name": "ActiDep Dataset",
    "BIDSVersion": "1.7.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": [
        "Nathan Decaux",
        "EMPENN Team"
    ],
    "Acknowledgements": "Data acquisition was supported by...",
    "HowToAcknowledge": "Please cite this dataset as...",
    "Funding": [
        "Grant from..."
    ],
    "ReferencesAndLinks": [
        "https://github.com/nathandecaux/ActiDep"
    ],
    "DatasetDOI": "doi:10.xxxx/xxxxx"
}
```

## Configuration Docker

### Dockerfile

```dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install FSL
RUN wget -O- https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py | python

# Install MRtrix3
RUN git clone https://github.com/MRtrix3/mrtrix3.git /opt/mrtrix3 \
    && cd /opt/mrtrix3 \
    && ./configure \
    && ./build

# Install ANIMA
RUN wget https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip \
    && unzip Anima-Ubuntu-4.2.zip -d /opt/ \
    && rm Anima-Ubuntu-4.2.zip

# Install TractSeg
RUN pip install TractSeg

# Install actiDep
COPY . /opt/actiDep
WORKDIR /opt/actiDep
RUN pip install -e .

# Set environment variables
ENV FSLDIR=/usr/local/fsl
ENV PATH="${FSLDIR}/bin:/opt/mrtrix3/bin:/opt/Anima-Binaries-4.2/bin:${PATH}"
ENV ANIMA_DIR=/opt/Anima-Binaries-4.2/bin

# Create configuration files
RUN mkdir -p ~/.anima ~/.tractseg-config \
    && echo "[anima-scripts]\nanima = /opt/Anima-Binaries-4.2/bin\nextra-data-root = /opt/Anima-Binaries-4.2/data\nanima-scripts-public-root = /opt/Anima-Binaries-4.2/scripts\nanima-scripts-root = /opt/Anima-Binaries-4.2/scripts" > ~/.anima/config.txt \
    && echo "[tractseg]\ntractseg-bin = /usr/local/bin\nmrtrix-bin = /opt/mrtrix3/bin" > ~/.tractseg-config/config.txt

WORKDIR /data
CMD ["python", "-c", "from actiDep.set_config import set_config; print('Configuration OK')"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  actidep:
    build: .
    container_name: actidep
    volumes:
      - ./data:/data
      - ./output:/output
      - ./temp:/tmp/actidep
    environment:
      - ACTIDEP_DATA=/data
      - ACTIDEP_OUTPUT=/output
      - ACTIDEP_TEMP=/tmp/actidep
      - ACTIDEP_NCORES=4
    working_dir: /data
    command: python -m actiDep.analysis.tractometry

  jupyter:
    build: .
    container_name: actidep-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./data:/data
      - ./output:/output
      - ./notebooks:/notebooks
    command: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

## Scripts de Configuration

### setup_environment.sh

Script d'installation automatique :

```bash
#!/bin/bash

echo "Configuration de l'environnement actiDep..."

# Vérification des prérequis
command -v python3 >/dev/null 2>&1 || { echo "Python 3 requis mais non installé. Abandon." >&2; exit 1; }

# Installation des dépendances Python
echo "Installation des dépendances Python..."
pip install -r requirements.txt

# Installation actiDep
echo "Installation actiDep..."
pip install -e .

# Création des dossiers de configuration
echo "Création des dossiers de configuration..."
mkdir -p ~/.anima ~/.tractseg-config ~/.cache/actidep

# Configuration ANIMA (si non existante)
if [ ! -f ~/.anima/config.txt ]; then
    echo "Configuration ANIMA..."
    read -p "Chemin vers ANIMA bin: " ANIMA_BIN
    read -p "Chemin vers ANIMA data: " ANIMA_DATA
    read -p "Chemin vers ANIMA scripts: " ANIMA_SCRIPTS
    
    cat > ~/.anima/config.txt << EOF
[anima-scripts]
anima = ${ANIMA_BIN}
extra-data-root = ${ANIMA_DATA}
anima-scripts-public-root = ${ANIMA_SCRIPTS}/public
anima-scripts-root = ${ANIMA_SCRIPTS}
EOF
fi

# Configuration TractSeg (si non existante)
if [ ! -f ~/.tractseg-config/config.txt ]; then
    echo "Configuration TractSeg..."
    read -p "Chemin vers TractSeg bin: " TRACTSEG_BIN
    read -p "Chemin vers MRtrix bin: " MRTRIX_BIN
    
    cat > ~/.tractseg-config/config.txt << EOF
[tractseg]
tractseg-bin = ${TRACTSEG_BIN}
mrtrix-bin = ${MRTRIX_BIN}
EOF
fi

# Test de la configuration
echo "Test de la configuration..."
python -c "
from actiDep.set_config import set_config
try:
    config, tools = set_config()
    print('✅ Configuration réussie!')
    print(f'ANIMA: {config[\"animaDir\"]}')
    print(f'MRtrix: {tools[\"mrtrix\"]}')
except Exception as e:
    print(f'❌ Erreur de configuration: {e}')
    exit(1)
"

echo "Configuration terminée avec succès!"
```

### validate_installation.py

Script de validation :

```python
#!/usr/bin/env python3
"""
Script de validation de l'installation actiDep.
"""

import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Vérifie les packages Python requis."""
    required_packages = [
        'numpy', 'pandas', 'nibabel', 'dipy', 
        'scipy', 'matplotlib', 'vtk', 'ants'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MANQUANT")
            missing.append(package)
    
    return missing

def check_external_tools():
    """Vérifie les outils externes."""
    tools = {
        'anima': ['animaDTIEstimator', '--help'],
        'mrtrix': ['tckgen', '--help'],
        'tractseg': ['TractSeg', '--help'],
        'fsl': ['fslinfo'],
        'ants': ['antsRegistration', '--help']
    }
    
    available = {}
    for tool, command in tools.items():
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ {tool}")
                available[tool] = True
            else:
                print(f"❌ {tool} - ERREUR")
                available[tool] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {tool} - NON TROUVÉ")
            available[tool] = False
    
    return available

def check_actidep():
    """Vérifie l'installation actiDep."""
    try:
        import actiDep
        from actiDep.set_config import set_config
        from actiDep.data.loader import Actidep, Subject
        
        print("✅ actiDep importé avec succès")
        
        # Test configuration
        config, tools = set_config()
        print("✅ Configuration chargée")
        
        return True
    except Exception as e:
        print(f"❌ actiDep - ERREUR: {e}")
        return False

def main():
    """Fonction principale de validation."""
    print("🔍 Validation de l'installation actiDep\n")
    
    print("📦 Vérification des packages Python:")
    missing_packages = check_python_packages()
    
    print("\n🛠️  Vérification des outils externes:")
    available_tools = check_external_tools()
    
    print("\n🐍 Vérification actiDep:")
    actidep_ok = check_actidep()
    
    print("\n📋 Résumé:")
    if missing_packages:
        print(f"❌ Packages Python manquants: {', '.join(missing_packages)}")
    else:
        print("✅ Tous les packages Python sont installés")
    
    unavailable_tools = [t for t, ok in available_tools.items() if not ok]
    if unavailable_tools:
        print(f"❌ Outils manquants: {', '.join(unavailable_tools)}")
    else:
        print("✅ Tous les outils externes sont disponibles")
    
    if actidep_ok:
        print("✅ actiDep est correctement installé et configuré")
    else:
        print("❌ Problème avec l'installation actiDep")
    
    # Code de sortie
    if missing_packages or unavailable_tools or not actidep_ok:
        print("\n⚠️  Installation incomplète")
        sys.exit(1)
    else:
        print("\n🎉 Installation complète et fonctionnelle!")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

## Configuration de Performance

### memory_config.py

Configuration pour optimiser l'utilisation mémoire :

```python
"""
Configuration mémoire pour actiDep.
"""

import psutil
import numpy as np

def get_optimal_config():
    """Retourne une configuration optimale selon les ressources système."""
    
    # Informations système
    total_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count()
    
    # Configuration adaptative
    config = {
        'n_cores': min(cpu_count - 1, 8),  # Laisser 1 CPU libre
        'memory_limit': int(total_memory * 0.7),  # 70% de la RAM
        'chunk_size': 1000,  # Streamlines par chunk
        'cache_size': int(total_memory * 0.1)  # 10% pour le cache
    }
    
    # Ajustements selon la mémoire disponible
    if total_memory < 8 * 1024**3:  # < 8 GB
        config.update({
            'n_cores': min(config['n_cores'], 4),
            'chunk_size': 500,
            'memory_limit': int(total_memory * 0.5)
        })
    elif total_memory > 32 * 1024**3:  # > 32 GB
        config.update({
            'chunk_size': 2000,
            'cache_size': int(total_memory * 0.2)
        })
    
    return config

# Configuration par défaut
PERFORMANCE_CONFIG = get_optimal_config()
```

Cette configuration complète permet d'adapter `actiDep` à différents environnements et besoins spécifiques.
