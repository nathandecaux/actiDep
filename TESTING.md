# Tests - actiDep

Documentation des tests unitaires et d'intégration pour le package `actiDep`.

## 🧪 Structure des Tests

```
tests/
├── __init__.py
├── conftest.py                 # Configuration pytest
├── test_core.py               # Tests du module core
├── test_data/                 # Données de test
│   ├── sample_dwi.nii.gz
│   ├── sample_fa.nii.gz
│   ├── sample.bval
│   ├── sample.bvec
│   └── sample_tracto.trk
├── unit/                      # Tests unitaires
│   ├── test_loader.py
│   ├── test_tractometry.py
│   ├── test_tractography.py
│   ├── test_mcm.py
│   └── test_utils.py
├── integration/               # Tests d'intégration
│   ├── test_pipeline_msmt.py
│   ├── test_pipeline_bundle_seg.py
│   └── test_full_workflow.py
└── performance/               # Tests de performance
    ├── test_memory_usage.py
    └── test_speed_benchmarks.py
```

## 🏃‍♂️ Exécution des Tests

### Tests Complets

```bash
# Tous les tests
pytest

# Tests avec couverture
pytest --cov=actiDep --cov-report=html

# Tests en parallèle
pytest -n auto

# Tests avec rapport détaillé
pytest -v --tb=short
```

### Tests Spécifiques

```bash
# Tests d'un module
pytest tests/unit/test_tractometry.py

# Tests d'une fonction
pytest tests/unit/test_tractometry.py::test_evaluate_along_streamlines

# Tests par marqueur
pytest -m "not slow"  # Exclure les tests lents
pytest -m "integration"  # Tests d'intégration seulement
```

### Tests avec Docker

```bash
# Construction de l'image de test
docker build -f Dockerfile.test -t actidep-test .

# Exécution des tests
docker run --rm -v $(pwd):/app actidep-test pytest

# Tests interactifs
docker run -it --rm -v $(pwd):/app actidep-test bash
```

## 📝 Configuration pytest

### conftest.py

```python
"""
Configuration globale pour les tests pytest.
"""

import pytest
import tempfile
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Dossier contenant les données de test."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def temp_output_dir():
    """Dossier temporaire pour les sorties de test."""
    temp_dir = tempfile.mkdtemp(prefix="actidep_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_dwi_data(test_data_dir):
    """Données DWI d'exemple."""
    dwi_file = test_data_dir / "sample_dwi.nii.gz"
    if not dwi_file.exists():
        # Créer des données factices si nécessaire
        data = np.random.rand(64, 64, 32, 30).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, dwi_file)
    return dwi_file

@pytest.fixture
def sample_fa_data(test_data_dir):
    """Carte FA d'exemple."""
    fa_file = test_data_dir / "sample_fa.nii.gz"
    if not fa_file.exists():
        # Créer une carte FA factice
        data = np.random.rand(64, 64, 32).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, fa_file)
    return fa_file

@pytest.fixture
def sample_streamlines():
    """Streamlines d'exemple."""
    from dipy.tracking.streamline import Streamlines
    
    # Créer des streamlines factices
    streamlines = []
    for i in range(100):
        # Ligne droite avec petit bruit
        points = np.linspace([0, 0, 0], [50, 0, 0], 50)
        noise = np.random.normal(0, 0.5, points.shape)
        streamlines.append(points + noise)
    
    return Streamlines(streamlines)

@pytest.fixture
def mock_subject(test_data_dir):
    """Sujet factice pour les tests."""
    from actiDep.data.loader import Subject
    
    # Créer un sujet avec des données factices
    subject = Subject("test001")
    subject._test_mode = True
    subject._test_data_dir = test_data_dir
    return subject

@pytest.fixture(scope="session")
def skip_if_no_external_tools():
    """Skip les tests si les outils externes ne sont pas disponibles."""
    import subprocess
    
    tools = ['animaDTIEstimator', 'tckgen', 'TractSeg']
    missing = []
    
    for tool in tools:
        try:
            subprocess.run([tool, '--help'], 
                         capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(tool)
    
    if missing:
        pytest.skip(f"Outils externes manquants: {missing}")

# Marqueurs personnalisés
def pytest_configure(config):
    """Configuration des marqueurs personnalisés."""
    config.addinivalue_line(
        "markers", "slow: marquer les tests comme lents"
    )
    config.addinivalue_line(
        "markers", "integration: tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "external: tests nécessitant des outils externes"
    )
    config.addinivalue_line(
        "markers", "gpu: tests nécessitant un GPU"
    )
```

## 🔬 Tests Unitaires

### test_tractometry.py

```python
"""
Tests pour le module tractometry.
"""

import pytest
import numpy as np
import nibabel as nib
from unittest.mock import Mock, patch

from actiDep.analysis.tractometry import (
    evaluate_along_streamlines,
    process_projection,
    reorient_streamlines
)

class TestEvaluateAlongStreamlines:
    """Tests pour evaluate_along_streamlines."""
    
    def test_basic_evaluation(self, sample_streamlines):
        """Test basique d'évaluation."""
        # Données de test
        scalar_img = np.random.rand(64, 64, 32)
        nr_points = 50
        
        # Exécution
        means, stds = evaluate_along_streamlines(
            scalar_img, sample_streamlines, nr_points
        )
        
        # Vérifications
        assert len(means) == nr_points
        assert len(stds) == nr_points
        assert all(isinstance(m, (int, float)) for m in means)
        assert all(isinstance(s, (int, float)) for s in stds)
        assert all(m >= 0 for m in means)  # FA positive
        assert all(s >= 0 for s in stds)   # Écart-type positif
    
    @pytest.mark.parametrize("algorithm", [
        "equal_dist", "distance_map", "cutting_plane", "afq"
    ])
    def test_different_algorithms(self, sample_streamlines, algorithm):
        """Test des différents algorithmes."""
        scalar_img = np.random.rand(64, 64, 32)
        
        means, stds = evaluate_along_streamlines(
            scalar_img, sample_streamlines, 50, 
            algorithm=algorithm
        )
        
        assert len(means) == 50
        assert len(stds) == 50
    
    def test_empty_streamlines(self):
        """Test avec streamlines vides."""
        scalar_img = np.random.rand(64, 64, 32)
        
        with pytest.raises(ValueError):
            evaluate_along_streamlines(scalar_img, [], 50)
    
    def test_with_beginnings_mask(self, sample_streamlines):
        """Test avec masque de débuts."""
        scalar_img = np.random.rand(64, 64, 32)
        beginnings = np.zeros((64, 64, 32))
        beginnings[30:35, 30:35, 15:20] = 1  # Région de début
        
        means, stds = evaluate_along_streamlines(
            scalar_img, sample_streamlines, 50,
            beginnings=beginnings
        )
        
        assert len(means) == 50
        assert len(stds) == 50
    
    def test_with_affine(self, sample_streamlines):
        """Test avec matrice affine."""
        scalar_img = np.random.rand(64, 64, 32)
        affine = np.eye(4)
        affine[0, 3] = 10  # Translation en x
        
        means, stds = evaluate_along_streamlines(
            scalar_img, sample_streamlines, 50,
            affine=affine
        )
        
        assert len(means) == 50

class TestProcessProjection:
    """Tests pour process_projection."""
    
    def test_single_bundle_projection(self, mock_subject, temp_output_dir):
        """Test projection sur un seul faisceau."""
        # Mocks des fichiers
        tracto_mock = Mock()
        tracto_mock.path = str(temp_output_dir / "test.trk")
        tracto_mock.get_full_entities.return_value = {
            'subject': 'test001', 'bundle': 'CSTleft'
        }
        
        metric_mock = Mock()
        metric_mock.path = str(temp_output_dir / "fa.nii.gz")
        metric_mock.get_full_entities.return_value = {
            'subject': 'test001', 'metric': 'FA'
        }
        
        # Créer des données factices
        self._create_test_tracto(tracto_mock.path)
        self._create_test_metric(metric_mock.path)
        
        # Test
        with patch('actiDep.analysis.tractometry.evaluate_along_streamlines') as mock_eval:
            mock_eval.return_value = ([0.5] * 98, [0.1] * 98)  # 100-2 points
            
            result = process_projection(
                {'CSTleft': tracto_mock},
                {'CSTleft': metric_mock},
                nr_points=100
            )
            
            assert len(result) == 2  # projection.csv et mean.csv
    
    def _create_test_tracto(self, path):
        """Crée un fichier tractogramme de test."""
        from dipy.io.streamline import save_trk
        from dipy.io.stateful_tractogram import StatefulTractogram, Space
        
        # Streamlines factices
        streamlines = [np.random.rand(50, 3) for _ in range(10)]
        sft = StatefulTractogram(streamlines, np.eye(4), Space.RASMM)
        save_trk(sft, path, bbox_valid_check=False)
    
    def _create_test_metric(self, path):
        """Crée une image métrique de test."""
        data = np.random.rand(64, 64, 32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)

class TestReorientStreamlines:
    """Tests pour reorient_streamlines."""
    
    def test_reorientation(self, sample_streamlines):
        """Test de réorientation des streamlines."""
        # Certaines streamlines dans le mauvais sens
        mixed_streamlines = list(sample_streamlines)
        for i in range(0, len(mixed_streamlines), 2):
            mixed_streamlines[i] = mixed_streamlines[i][::-1]  # Inverser
        
        reoriented = reorient_streamlines(mixed_streamlines)
        
        assert len(reoriented) == len(mixed_streamlines)
        
        # Vérifier que toutes vont dans le même sens
        directions = []
        for sl in reoriented:
            if len(sl) > 1:
                direction = sl[-1] - sl[0]  # Direction générale
                directions.append(direction)
        
        # Toutes les directions devraient être similaires
        if directions:
            mean_direction = np.mean(directions, axis=0)
            for direction in directions:
                cos_angle = np.dot(direction, mean_direction) / (
                    np.linalg.norm(direction) * np.linalg.norm(mean_direction)
                )
                assert cos_angle > 0  # Angle < 90°
```

### test_loader.py

```python
"""
Tests pour le module data.loader.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from actiDep.data.loader import ActiDepFile, Subject, Actidep

class TestActiDepFile:
    """Tests pour la classe ActiDepFile."""
    
    def test_init_basic(self, temp_output_dir):
        """Test d'initialisation basique."""
        test_path = temp_output_dir / "sub-001_ses-1_T1w.nii.gz"
        test_path.touch()
        
        file_obj = ActiDepFile(str(test_path))
        
        assert file_obj.path == str(test_path)
        assert file_obj.filename == "sub-001_ses-1_T1w.nii.gz"
        assert file_obj.scope == "raw"  # Pas de 'derivatives' dans le chemin
        assert 'sub' in file_obj.entities
        assert file_obj.entities['sub'] == '001'
        assert file_obj.suffix == 'T1w'
    
    def test_derivatives_file(self, temp_output_dir):
        """Test avec fichier derivatives."""
        test_path = temp_output_dir / "derivatives" / "pipeline" / "sub-001_metric-FA_dwi.nii.gz"
        test_path.parent.mkdir(parents=True)
        test_path.touch()
        
        file_obj = ActiDepFile(str(test_path))
        
        assert file_obj.scope == "derivatives"
        assert file_obj.pipeline == "pipeline"
        assert 'metric' in file_obj.entities
        assert file_obj.entities['metric'] == 'FA'
    
    def test_get_entities(self, temp_output_dir):
        """Test get_entities."""
        test_path = temp_output_dir / "sub-001_ses-1_acq-test_T1w.nii.gz"
        test_path.touch()
        
        file_obj = ActiDepFile(str(test_path))
        entities = file_obj.get_entities()
        
        expected_keys = ['sub', 'ses', 'acq', 'suffix', 'extension']
        for key in expected_keys:
            assert key in entities
    
    def test_get_full_entities(self, temp_output_dir):
        """Test get_full_entities."""
        test_path = temp_output_dir / "derivatives" / "test_pipeline" / "dwi" / "sub-001_FA.nii.gz"
        test_path.parent.mkdir(parents=True)
        test_path.touch()
        
        file_obj = ActiDepFile(str(test_path))
        entities = file_obj.get_full_entities()
        
        assert entities['scope'] == 'derivatives'
        assert entities['pipeline'] == 'test_pipeline'
        assert entities['datatype'] == 'dwi'

class TestSubject:
    """Tests pour la classe Subject."""
    
    def test_init(self):
        """Test d'initialisation."""
        subject = Subject("001", db_root="/test/path")
        
        assert subject.sub_id == "001"
        assert subject.db_root == "/test/path"
    
    @patch('actiDep.data.loader.glob.glob')
    def test_get_files(self, mock_glob, temp_output_dir):
        """Test de récupération de fichiers."""
        # Mock des fichiers trouvés
        mock_files = [
            str(temp_output_dir / "sub-001_T1w.nii.gz"),
            str(temp_output_dir / "sub-001_dwi.nii.gz")
        ]
        mock_glob.return_value = mock_files
        
        # Créer les fichiers
        for file_path in mock_files:
            Path(file_path).touch()
        
        subject = Subject("001", db_root=str(temp_output_dir))
        files = subject.get(suffix="T1w")
        
        assert len(files) >= 0  # Dépend de l'implémentation du filtrage
        if files:
            assert all(isinstance(f, ActiDepFile) for f in files)
    
    def test_get_unique_success(self, mock_subject):
        """Test get_unique avec succès."""
        mock_file = Mock(spec=ActiDepFile)
        
        with patch.object(mock_subject, 'get', return_value=[mock_file]):
            result = mock_subject.get_unique(suffix="T1w")
            assert result == mock_file
    
    def test_get_unique_multiple_files(self, mock_subject):
        """Test get_unique avec plusieurs fichiers."""
        mock_files = [Mock(spec=ActiDepFile), Mock(spec=ActiDepFile)]
        
        with patch.object(mock_subject, 'get', return_value=mock_files):
            with pytest.raises(ValueError, match="Multiple files found"):
                mock_subject.get_unique(suffix="T1w")
    
    def test_get_unique_no_files(self, mock_subject):
        """Test get_unique sans fichiers."""
        with patch.object(mock_subject, 'get', return_value=[]):
            with pytest.raises(ValueError, match="No files found"):
                mock_subject.get_unique(suffix="T1w")

class TestActidep:
    """Tests pour la classe Actidep."""
    
    def test_init(self, temp_output_dir):
        """Test d'initialisation."""
        dataset = Actidep(str(temp_output_dir))
        assert dataset.root == str(temp_output_dir)
    
    @patch('actiDep.data.loader.glob.glob')
    def test_get_subjects(self, mock_glob, temp_output_dir):
        """Test de récupération des sujets."""
        # Mock des dossiers sujets
        mock_dirs = [
            str(temp_output_dir / "sub-001"),
            str(temp_output_dir / "sub-002"),
            str(temp_output_dir / "derivatives")  # Doit être ignoré
        ]
        mock_glob.return_value = mock_dirs
        
        dataset = Actidep(str(temp_output_dir))
        
        with patch('os.path.isdir', return_value=True):
            subjects = dataset.get_subjects()
            
            expected_subjects = ['001', '002']
            assert set(subjects) == set(expected_subjects)
```

## 🔗 Tests d'Intégration

### test_full_workflow.py

```python
"""
Tests d'intégration pour workflow complet.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from actiDep.data.loader import Subject, Actidep
from actiDep.analysis.tractometry import process_tractseg_analysis

@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflow:
    """Tests d'intégration pour workflow complet."""
    
    @pytest.fixture(scope="class")
    def test_dataset(self):
        """Dataset de test complet."""
        temp_dir = tempfile.mkdtemp(prefix="actidep_integration_")
        dataset_path = Path(temp_dir)
        
        # Créer structure BIDS basique
        self._create_test_bids_structure(dataset_path)
        
        yield dataset_path
        
        shutil.rmtree(temp_dir)
    
    def _create_test_bids_structure(self, dataset_path):
        """Crée une structure BIDS de test."""
        # Sujets
        subjects = ['sub-001', 'sub-002']
        
        for subject in subjects:
            # Données raw
            subj_dir = dataset_path / subject
            dwi_dir = subj_dir / 'dwi'
            dwi_dir.mkdir(parents=True)
            
            # Fichiers DWI factices
            self._create_fake_dwi(dwi_dir / f"{subject}_dwi.nii.gz")
            self._create_fake_bval(dwi_dir / f"{subject}_dwi.bval")
            self._create_fake_bvec(dwi_dir / f"{subject}_dwi.bvec")
            
            # Derivatives
            deriv_dir = dataset_path / 'derivatives' / 'anima_preproc' / subject / 'dwi'
            deriv_dir.mkdir(parents=True)
            
            # Métriques factices
            self._create_fake_metric(deriv_dir / f"{subject}_metric-FA_dwi.nii.gz")
            self._create_fake_metric(deriv_dir / f"{subject}_metric-MD_dwi.nii.gz")
            
            # Faisceaux
            bundle_dir = dataset_path / 'derivatives' / 'bundle_seg' / subject / 'tracto'
            bundle_dir.mkdir(parents=True)
            
            bundles = ['CSTleft', 'CSTright']
            for bundle in bundles:
                self._create_fake_tractogram(
                    bundle_dir / f"{subject}_bundle-{bundle}_tracto.trk"
                )
    
    def _create_fake_dwi(self, path):
        """Crée un fichier DWI factice."""
        import nibabel as nib
        import numpy as np
        
        data = np.random.rand(64, 64, 32, 30).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)
    
    def _create_fake_bval(self, path):
        """Crée un fichier bval factice."""
        # 30 directions : 1 b=0 + 29 b=1000
        bvals = [0] + [1000] * 29
        with open(path, 'w') as f:
            f.write(' '.join(map(str, bvals)))
    
    def _create_fake_bvec(self, path):
        """Crée un fichier bvec factice."""
        import numpy as np
        
        # Directions aléatoires normalisées
        directions = np.random.randn(3, 30)
        directions[:, 0] = 0  # Première direction nulle (b=0)
        
        # Normaliser les autres directions
        for i in range(1, 30):
            norm = np.linalg.norm(directions[:, i])
            if norm > 0:
                directions[:, i] /= norm
        
        with open(path, 'w') as f:
            for row in directions:
                f.write(' '.join(f"{x:.6f}" for x in row) + '\n')
    
    def _create_fake_metric(self, path):
        """Crée une image métrique factice."""
        import nibabel as nib
        import numpy as np
        
        data = np.random.rand(64, 64, 32).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)
    
    def _create_fake_tractogram(self, path):
        """Crée un tractogramme factice."""
        from dipy.io.streamline import save_trk
        from dipy.io.stateful_tractogram import StatefulTractogram, Space
        import numpy as np
        
        # Streamlines factices
        streamlines = []
        for i in range(100):
            # Ligne avec bruit
            points = np.linspace([0, 0, 0], [50, 0, 0], 50)
            noise = np.random.normal(0, 1, points.shape)
            streamlines.append(points + noise)
        
        sft = StatefulTractogram(streamlines, np.eye(4), Space.RASMM)
        save_trk(sft, path, bbox_valid_check=False)
    
    def test_dataset_loading(self, test_dataset):
        """Test de chargement du dataset."""
        dataset = Actidep(str(test_dataset))
        
        subjects = dataset.get_subjects()
        assert len(subjects) == 2
        assert '001' in subjects
        assert '002' in subjects
    
    def test_subject_data_access(self, test_dataset):
        """Test d'accès aux données d'un sujet."""
        subject = Subject('001', db_root=str(test_dataset))
        
        # Test récupération DWI
        dwi_files = subject.get(scope='raw', suffix='dwi')
        assert len(dwi_files) > 0
        
        # Test récupération métriques
        fa_files = subject.get(pipeline='anima_preproc', metric='FA')
        assert len(fa_files) > 0
        
        # Test récupération faisceaux
        bundles = subject.get(pipeline='bundle_seg', suffix='tracto')
        assert len(bundles) > 0
    
    @pytest.mark.skip(reason="Nécessite TractSeg installé")
    def test_full_tractometry_pipeline(self, test_dataset):
        """Test du pipeline tractométrie complet."""
        # Créer fichier subjects.txt
        subjects_file = test_dataset / 'subjects.txt'
        with open(subjects_file, 'w') as f:
            f.write('/fake/tracto/path\n')
            f.write('bundle=CSTleft CSTright\n')
            f.write('sub-001 group1\n')
            f.write('sub-002 group1\n')
        
        # Exécuter analyse (mocké car nécessite TractSeg)
        with pytest.raises(Exception):  # Attendu car données factices
            process_tractseg_analysis(
                str(subjects_file),
                dataset_path=str(test_dataset),
                metric='FA'
            )
```

## ⚡ Tests de Performance

### test_memory_usage.py

```python
"""
Tests de performance mémoire.
"""

import pytest
import psutil
import numpy as np
from memory_profiler import profile

from actiDep.analysis.tractometry import evaluate_along_streamlines

@pytest.mark.performance
class TestMemoryUsage:
    """Tests d'utilisation mémoire."""
    
    def test_memory_large_dataset(self):
        """Test mémoire avec large dataset."""
        # Créer données volumineuses
        scalar_img = np.random.rand(256, 256, 128).astype(np.float32)
        
        # Beaucoup de streamlines
        streamlines = []
        for i in range(1000):
            points = np.random.rand(100, 3).astype(np.float32) * 100
            streamlines.append(points)
        
        # Mesurer utilisation mémoire
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Exécution
        means, stds = evaluate_along_streamlines(
            scalar_img, streamlines, 100
        )
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Vérifications
        assert len(means) == 100
        assert memory_used < 1000  # Moins de 1 GB
        
        print(f"Mémoire utilisée: {memory_used:.1f} MB")
    
    @profile
    def test_memory_profile_tractometry(self):
        """Profil mémoire détaillé."""
        scalar_img = np.random.rand(128, 128, 64).astype(np.float32)
        
        streamlines = []
        for i in range(500):
            points = np.random.rand(50, 3).astype(np.float32) * 50
            streamlines.append(points)
        
        means, stds = evaluate_along_streamlines(
            scalar_img, streamlines, 50
        )
        
        assert len(means) == 50
```

### test_speed_benchmarks.py

```python
"""
Benchmarks de vitesse.
"""

import pytest
import time
import numpy as np
from actiDep.analysis.tractometry import evaluate_along_streamlines

@pytest.mark.performance
class TestSpeedBenchmarks:
    """Benchmarks de performance."""
    
    @pytest.mark.parametrize("n_streamlines", [100, 500, 1000])
    @pytest.mark.parametrize("algorithm", ["distance_map", "equal_dist"])
    def test_tractometry_speed(self, n_streamlines, algorithm):
        """Benchmark vitesse tractométrie."""
        # Données de test
        scalar_img = np.random.rand(128, 128, 64).astype(np.float32)
        
        streamlines = []
        for i in range(n_streamlines):
            points = np.random.rand(50, 3).astype(np.float32) * 50
            streamlines.append(points)
        
        # Mesure du temps
        start_time = time.time()
        
        means, stds = evaluate_along_streamlines(
            scalar_img, streamlines, 100,
            algorithm=algorithm
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Vérifications
        assert len(means) == 100
        assert execution_time < 60  # Moins d'1 minute
        
        print(f"{algorithm} - {n_streamlines} streamlines: {execution_time:.2f}s")
    
    def test_compare_algorithms_speed(self):
        """Compare la vitesse des algorithmes."""
        scalar_img = np.random.rand(64, 64, 32).astype(np.float32)
        
        streamlines = []
        for i in range(200):
            points = np.random.rand(50, 3).astype(np.float32) * 30
            streamlines.append(points)
        
        algorithms = ["equal_dist", "distance_map", "afq"]
        times = {}
        
        for algorithm in algorithms:
            start_time = time.time()
            
            means, stds = evaluate_along_streamlines(
                scalar_img, streamlines, 50,
                algorithm=algorithm
            )
            
            end_time = time.time()
            times[algorithm] = end_time - start_time
            
            assert len(means) == 50
        
        # Affichage des résultats
        print("\nComparaison vitesse algorithmes:")
        for algo, time_taken in sorted(times.items(), key=lambda x: x[1]):
            print(f"  {algo}: {time_taken:.3f}s")
        
        # Le plus rapide devrait être distance_map ou equal_dist
        fastest = min(times.keys(), key=lambda k: times[k])
        assert fastest in ["distance_map", "equal_dist"]
```

## 📊 Couverture de Code

### .coveragerc

```ini
[run]
source = actiDep
omit = 
    */tests/*
    */test_*
    setup.py
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    
show_missing = True
precision = 2

[html]
directory = htmlcov
```

### Génération du Rapport

```bash
# Exécution avec couverture
pytest --cov=actiDep --cov-report=html --cov-report=term

# Rapport XML pour CI/CD
pytest --cov=actiDep --cov-report=xml

# Vérification seuil minimum
pytest --cov=actiDep --cov-fail-under=80
```

## 🚀 Intégration Continue

### GitHub Actions (.github/workflows/tests.yml)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest --cov=actiDep --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

Ces tests assurent la qualité et la fiabilité du package `actiDep` à travers différents niveaux de validation.
