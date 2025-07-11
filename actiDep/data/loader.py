import os
import glob
import json
import shutil
import pathlib
import nibabel as nib
import SimpleITK as sitk
import re  # Ajout du module re pour les expressions régulières
from os.path import join as opj
from actiDep.utils.tools import create_pipeline_description
from actiDep.utils.hash_utils import check_folder_changed, save_folder_hash
from actiDep.data.io import copy2nii, move2nii, parse_filename
import threading

_SENTINEL = object()  # Défini au niveau du module

class ActiDepFile:
    """
    Une classe pour représenter un fichier dans la base de données BIDS.
    Cette classe est un remplaçant léger de BIDSFile de PyBIDS.
    """
    def __init__(self, path, subject=None):
        self.path = path
        self.filename = os.path.basename(path)
        self.subject_obj = subject
        
        # Déterminer si le fichier est dans le scope 'derivatives' ou 'raw'
        self.scope = 'derivatives' if 'derivatives' in path else 'raw'
        
        # Extraire le pipeline si applicable
        self.pipeline = None
        if self.scope == 'derivatives':
            parts = path.split('derivatives/')[1].split('/')
            if len(parts) > 0:
                self.pipeline = parts[0]
        
        # Déterminer le datatype (dossier parent)
        self.datatype = None
        path_parts = os.path.dirname(path).split('/')
        if len(path_parts) > 0:
            self.datatype = path_parts[-1]
        
        # Extraire les entités du nom de fichier
        self.entities = parse_filename(self.filename)
        
        # Ajouter l'entité 'subject'
        # Si self.subject_obj existe, utiliser son sub_id, sinon essayer de l'extraire des entités ou du chemin
        if self.subject_obj is not None and hasattr(self.subject_obj, 'sub_id'):
            self.entities['subject'] = self.subject_obj.sub_id
        elif 'sub' in self.entities:
            self.entities['subject'] = self.entities['sub']
        else:
            # Essayer d'extraire l'ID du sujet du chemin (ex: .../sub-XXXXX/...)
            match = re.search(r'sub-(\w+)', path)
            if match:
                self.entities['subject'] = match.group(1)
        
        # Ajouter l'extension comme entité supplémentaire, en s'assurant qu'elle commence par '.'
        # et gère correctement les extensions composées (ex: .nii.gz).
        if '.' in self.filename:
            first_dot_index = self.filename.find('.')
            self.entities['extension'] = self.filename[first_dot_index:]
        # Si le nom de fichier n'a pas de point, l'entité 'extension' n'est pas définie ou modifiée par ce bloc.
        
        # Extraire explicitement le suffix si présent dans le nom de fichier
        if '_' in self.filename:
            parts = self.filename.split('_')
            last_part = parts[-1].split('.')[0]  # Enlever l'extension si présente
            # Vérifier si le dernier segment n'est pas une entité (qui aurait la forme key-value)
            if '-' not in last_part:
                self.entities['suffix'] = last_part
        
        # Supprimer l'entité 'sub' des entités supplémentaires
        self.extra_entities = dict(self.entities)
        self.extra_entities.pop('sub', None)
        
        # Définir les entités comme attributs
        for k, v in self.get_full_entities().items():
            setattr(self, k, v)
    
    def copy(self, dest):
        """Copier le fichier vers une nouvelle destination, en convertissant si nécessaire."""
        return copy2nii(self.path, dest)
    
    def move(self, dest):
        """Déplacer le fichier vers une nouvelle destination, en convertissant si nécessaire."""
        return move2nii(self.path, dest)
    
    def get_entities(self, **kwargs):
        """
        Obtenir les entités de base du fichier.
        Compatibilité avec l'API PyBIDS.
        """
        return self.entities
    
    def get_full_entities(self, **kwargs):
        """
        Obtenir toutes les entités du fichier, y compris le scope, le pipeline et le datatype.
        """
        entities = dict(self.entities)
        entities['scope'] = self.scope
        entities['pipeline'] = self.pipeline
        if self.datatype:
            entities['datatype'] = self.datatype
        entities.update(self.extra_entities)
        return entities
    
    def get_subject(self):
        """
        Retourner l'objet Subject associé à ce fichier.
        """
        if self.subject_obj is not None:
            return self.subject_obj
        else:
            return Subject(self.entities.get('sub'))
    
    def __str__(self):
        print(self.path+'\n')
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.path == other
        elif isinstance(other, ActiDepFile):
            return self.path == other.path
        return False
    
    def __hash__(self):
        """
        Implémentation de __hash__ pour rendre ActiDepFile hashable.
        La valeur de hachage est basée sur le chemin du fichier.
        """
        return hash(self.path)


class Subject:
    """
    Une classe pour représenter un sujet dans la base de données BIDS.
    Cette classe est un remplaçant léger du Subject de PyBIDS.
    """
    def __init__(self, sub_id, db_root='/home/ndecaux/Data/actidep_bids', layout=None, force_index=False):
        self.db_root = db_root
        self.sub_id = sub_id
        self.bids_id = f'sub-{sub_id}'
        self.hash_file_path = opj(os.path.expanduser('~'), f'.actidep_hash_{sub_id}.json')
        
        # Vérifier si le dossier a été modifié
        folder_changed, _ = check_folder_changed(db_root, self.hash_file_path)
        
        # Initialiser le cache des fichiers et les index d'entités
        self.files_cache = {}
        self.entity_indices = {} # Ajout pour stocker les index d'entités
        
        # Indexer les fichiers si nécessaire
        if force_index or folder_changed or not os.path.exists(self.hash_file_path): # Vérifier aussi si le hash_file existe
            self._index_files()
            save_folder_hash(db_root, self.hash_file_path, {'subject': sub_id, 'last_indexed': True})
        elif not self.files_cache: # Si le cache est vide mais le hash existe et est à jour (cas rare, ex: premier chargement après init)
             self._index_files() # Forcer l'indexation pour remplir le cache mémoire
        
        # Dossier DICOM pour ce sujet
        self.dicom_folder = opj(self.db_root, self.bids_id, 'sourcedata', f'{self.bids_id}_dicoms')
    
    def _index_files(self):
        """
        Indexer tous les fichiers du sujet dans la base BIDS.
        Utilise os.walk pour une meilleure performance et peuple les entity_indices.
        """
        self.files_cache = {}
        self.entity_indices = {} # Réinitialiser les index à chaque réindexation

        def process_tree(root_path):
            if not os.path.isdir(root_path):
                return
            
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
                if 'sourcedata' in dirnames:
                    dirnames.remove('sourcedata')
                
                if 'sourcedata' in dirpath.split(os.sep):
                    dirnames[:] = [] 
                    continue

                for filename in filenames:
                    if not filename.startswith('.') and not filename.endswith('.DS_Store'):
                        filepath = opj(dirpath, filename)
                        file_obj = ActiDepFile(filepath, self)
                        self.files_cache[filepath] = file_obj

                        # Peupler entity_indices
                        for entity_key, entity_val in file_obj.get_full_entities().items():
                            self.entity_indices.setdefault(entity_key, {}).setdefault(entity_val, set()).add(file_obj)

        subject_main_path = opj(self.db_root, self.bids_id)
        process_tree(subject_main_path)

        derivatives_root_path = opj(self.db_root, 'derivatives')
        if os.path.isdir(derivatives_root_path):
            for pipeline_name in os.listdir(derivatives_root_path):
                pipeline_dir = opj(derivatives_root_path, pipeline_name)
                if os.path.isdir(pipeline_dir):
                    subject_derivative_path = opj(pipeline_dir, self.bids_id)
                    process_tree(subject_derivative_path)
    
    def refresh(self):
        """
        Forcer la réindexation des fichiers pour prendre en compte les changements.
        """
        folder_changed, old_hash = check_folder_changed(self.db_root, self.hash_file_path)
        
        if not folder_changed:
            # Message mis à jour pour inclure l'ID du sujet
            print(f"Le dossier {self.db_root} n'a pas été modifié depuis la dernière indexation pour le sujet {self.sub_id}")
            return
        
        self._index_files()
        save_folder_hash(self.db_root, self.hash_file_path, {'subject': self.sub_id, 'last_indexed': True})
    
    def get(self, **kwargs):
        """
        Récupérer les fichiers qui correspondent aux critères donnés, en utilisant les index d'entités.
        """
        if not self.files_cache: # Si le cache est vide (par exemple, après un init sans force_index mais avec un hash à jour)
            self._index_files() 
            if not self.files_cache and kwargs: 
                return []

        normalized_kwargs = {}
        for k_kwarg, v_kwarg_orig in kwargs.items():
            v_kwarg = v_kwarg_orig
            if k_kwarg == 'extension' and isinstance(v_kwarg_orig, str):
                if v_kwarg_orig.startswith('!'):
                    if len(v_kwarg_orig) > 1 and v_kwarg_orig[1] != '.':
                        v_kwarg = '!.' + v_kwarg_orig[1:]
                elif not v_kwarg_orig.startswith('.'):
                    v_kwarg = '.' + v_kwarg_orig
            normalized_kwargs[k_kwarg] = v_kwarg

        if not normalized_kwargs:
            return list(self.files_cache.values())

        candidate_files = None
        indexed_positive_kwargs = {}
        complex_kwargs = {}

        for k, v in normalized_kwargs.items():
            if isinstance(v, str) and not v.startswith('!') and k in self.entity_indices:
                indexed_positive_kwargs[k] = v
            else:
                complex_kwargs[k] = v
        
        for k_crit, v_crit in indexed_positive_kwargs.items():
            current_match_set = self.entity_indices.get(k_crit, {}).get(v_crit, set())
            if candidate_files is None:
                candidate_files = current_match_set.copy()
            else:
                candidate_files.intersection_update(current_match_set)
            
            if not candidate_files:
                return []

        if candidate_files is None: # Aucun kwarg positif indexé n'a été traité ou n'a correspondu
            candidate_files = set(self.files_cache.values())


        if not complex_kwargs:
            return list(candidate_files)

        # Appliquer les filtres complexes
        final_files_list = []
        for file_obj in candidate_files:
            match = True
            for k_crit, v_crit in complex_kwargs.items():
                file_entity_value = getattr(file_obj, k_crit, _SENTINEL)

                if v_crit is None:
                    if file_entity_value is not _SENTINEL:
                        match = False
                        break
                elif isinstance(v_crit, str) and v_crit.startswith('!'):
                    excluded_value = v_crit[1:]
                    if file_entity_value is not _SENTINEL and file_entity_value == excluded_value:
                        match = False
                        break
                # Cas pour les entités non indexées positivement (rare si l'indexation est complète)
                # ou si un type de valeur non-str a été passé pour un critère non-extension.
                elif file_entity_value is _SENTINEL or file_entity_value != v_crit:
                    match = False
                    break
            
            if match:
                final_files_list.append(file_obj)
        
        return final_files_list
    
    def get_unique(self, **kwargs):
        """
        Récupérer le fichier unique qui correspond aux critères donnés.
        """
        files = self.get(**kwargs)
        if len(files) == 0:
            raise ValueError(f"Aucun fichier trouvé pour {kwargs}")
        if len(files) > 1:
            raise ValueError(f"Plusieurs fichiers trouvés pour {kwargs}")
        return files[0]
    
    
    def build_path(self, suffix, pipeline=None, scope=None, original_name=None, is_dir=False, **kwargs):
        """
        Construire le chemin d'un fichier dans le layout à partir d'entités et d'un pipeline donnés.
        """
        entities = {'suffix': suffix}
        entities.update(kwargs) # Mettre à jour avec kwargs, permettant à kwargs de surcharger le suffixe

        root_dir = self.db_root
        
        # Déterminer parent_dir (correspondant au répertoire du datatype BIDS)
        # L'entité 'datatype' explicite a la priorité
        parent_dir = entities.pop('datatype', None)

        if parent_dir is None: # Si datatype n'a été explicitement fourni
            if pipeline is not None:
                # Pour les dérivés: si 'datatype' n'est pas spécifié, utiliser 'dwi' par défaut.
                parent_dir = 'dwi'
                # print(f"Warning: Construction du chemin pour un dérivé (pipeline='{pipeline}') sans 'datatype' explicite. Utilisation par défaut de '{parent_dir}'. Entités: {entities}")
            elif 'suffix' in entities:
                # Pour les données brutes (pipeline is None): utiliser le suffixe comme datatype.
                parent_dir = entities['suffix']
            else:
                # Ce cas ne devrait pas être atteint si les entités contiennent toujours au moins un suffixe.
                raise ValueError("Impossible de déterminer le 'datatype' BIDS (parent_dir): "
                                 "l'entité 'datatype' est manquante et le 'suffix' n'est pas disponible.")

        if pipeline is None:
            target_dir = opj(root_dir, self.bids_id, parent_dir)
        else:
            # C'est un dérivé
            pipeline_desc_dir = opj(root_dir, 'derivatives', pipeline)
            if not os.path.exists(pipeline_desc_dir):
                os.makedirs(pipeline_desc_dir, exist_ok=True)
                pipeline_desc_file = opj(pipeline_desc_dir, 'dataset_description.json')
                if not os.path.exists(pipeline_desc_file):
                    # Assurez-vous que create_pipeline_description peut gérer layout=None ou passez self.layout si disponible et pertinent
                    create_pipeline_description(pipeline, None) 
            
            target_dir = opj(root_dir, 'derivatives', pipeline, self.bids_id, parent_dir)
        
        fileradix = f'{self.bids_id}'
        
        # Déterminer l'extension
        current_extension = entities.pop('extension', None) # Retirer l'extension des entités pour la gestion manuelle

        if original_name is not None and current_extension is None:
            name_parts = original_name.split('.')
            if len(name_parts) > 2 and name_parts[-2].lower() == 'nii' and name_parts[-1].lower() == 'gz':
                extension = '.nii.gz'
            elif len(name_parts) > 1:
                extension = '.' + name_parts[-1]
            else: # Cas où original_name n'a pas d'extension claire (ex: "monfichier")
                extension = '.nii.gz' # Par défaut pour les images, ou '' si is_dir est vrai plus tard
                # print(f"Warning: No extension in entities and cannot reliably determine from original_name '{original_name}', assuming {extension} by default for non-directory.")
        elif current_extension is not None:
            extension = '.' + current_extension.lstrip('.')
        else: # Ni original_name pour déduire, ni extension explicite
            is_dir_flag = entities.get('is_dir', is_dir)
            extension = '' if is_dir_flag else '.nii.gz'
            if not is_dir_flag:
                 print(f"Warning: No extension provided or deducible for {entities}, assuming {extension}")
        
        # Ajouter les entités restantes au nom du fichier
        # Exclure 'suffix', 'subject', 'sub', 'is_dir' car déjà gérés ou non pertinents pour le nom de fichier ici
        # 'datatype' et 'extension' ont déjà été retirés/gérés
        remaining_entities = {
            k: v
            for k, v in entities.items()
            if k not in ['suffix', 'subject', 'sub', 'is_dir'] 
        }
        
        for k, v in sorted(remaining_entities.items()):
            fileradix += f'_{k}-{v}'
        
        # Le suffixe est la dernière partie descriptive du nom de fichier avant l'extension
        path = opj(target_dir, fileradix + "_" + entities['suffix'] + extension)
        
        return path
    
    def get_pipelines(self):
        """
        Retourner la liste des pipelines qui ont été exécutées sur ce sujet.
        """
        pipelines = set()
        for filepath in self.files_cache.keys():
            if 'derivatives' in filepath:
                parts = filepath.split('derivatives/')[1].split('/')
                if len(parts) > 0:
                    pipelines.add(parts[0])
        return list(pipelines)
    
    def write_object(self, obj, suffix, pipeline=None, scope=None, original_name=None, refresh_layout=True, **kwargs):
        """
        Écrire un objet dans un fichier dans le layout.
        """
        path = self.build_path(suffix, pipeline, scope, original_name, **kwargs)
        
        # Créer le répertoire si nécessaire
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        
        # Écrire l'objet selon son type
        if isinstance(obj, sitk.Image):
            sitk.WriteImage(obj, path)
        elif hasattr(obj, 'to_file'):  # Pour les objets ants.ANTsImage
            obj.to_file(path)
        elif isinstance(obj, dict):
            with open(path, 'w') as f:
                json.dump(obj, f)
        elif isinstance(obj, nib.Nifti1Image):
            nib.save(obj, path)
        elif isinstance(obj, str):
            copy2nii(obj, path)
        else:
            raise ValueError(f"Type d'objet {type(obj)} non pris en charge pour l'écriture.")
        
        # Mettre à jour le hash après l'écriture
        save_folder_hash(self.db_root, self.hash_file_path, {'subject': self.sub_id, 'last_write': True})
        
        # Réindexer si demandé
        if refresh_layout:
            self.refresh()
        
        return path
    
    def get_entity_values(self):
        """
        Obtenir toutes les entités qui existent pour ce sujet et leurs valeurs possibles.
        """
        entity_values = {}
        
        # Indexer les fichiers si le cache est vide
        if not self.files_cache:
            self._index_files()
        
        # Exclure les fichiers .dcm
        files = [f for f in self.files_cache.values() if not f.path.endswith('.dcm')]
        
        # Traiter chaque fichier
        for file in files:
            # Obtenir les entités du fichier
            entities = file.get_full_entities()
            
            # Traiter chaque entité
            for entity, value in entities.items():
                if entity == 'subject':  # Ignorer subject car constant pour cette instance
                    continue
                
                # Initialiser la liste pour cette entité si elle n'existe pas
                if entity not in entity_values:
                    entity_values[entity] = []
                
                # Ajouter la valeur si elle n'est pas déjà dans la liste
                if value not in entity_values[entity]:
                    entity_values[entity].append(value)
        
        # Trier les valeurs
        for entity in entity_values:
            entity_values[entity].sort()
        
        return entity_values


class Actidep:
    """
    Une classe pour représenter la base de données BIDS complète.
    Cette classe est un remplaçant léger de BIDSLayout de PyBIDS.
    """
    def __init__(self, db_root='/home/ndecaux/Data/actidep_bids', force_index=False):
        self.db_root = db_root
        self.hash_file_path = opj(os.path.expanduser('~'), '.actidep_hash_global.json')
        self._subjects_cache = {}  # Cache pour les objets Subject
        
        # Vérifier si le dossier a été modifié
        folder_changed, _ = check_folder_changed(db_root, self.hash_file_path)
        
        # Indexer les sujets si nécessaire
        self.subject_ids = []
        self._index_subjects()
        
        # Sauvegarder le hash après indexation
        if folder_changed or force_index:
            save_folder_hash(db_root, self.hash_file_path, {'global_db': True})
    
    def _index_subjects(self):
        """
        Indexer tous les sujets dans la base BIDS.
        """
        subject_ids_set = set()  # Utiliser un ensemble pour une collecte efficace et unique
        
        # Trouver tous les dossiers qui commencent par 'sub-'
        subject_pattern = opj(self.db_root, 'sub-*')
        for dirpath in glob.glob(subject_pattern):
            if os.path.isdir(dirpath):
                dirname = os.path.basename(dirpath)
                # Vérifier que le nom commence bien par 'sub-'
                if dirname.startswith('sub-'):
                    sub_id = dirname.replace('sub-', '')
                    # Vérifier que l'ID du sujet est bien un nombre entier
                    if re.match(r'^\d+$', sub_id):
                        subject_ids_set.add(sub_id)
        
        # Trouver aussi les sujets qui n'existent que dans les dérivés
        derivatives_pattern = opj(self.db_root, 'derivatives', '*', 'sub-*')
        for dirpath in glob.glob(derivatives_pattern):
            if os.path.isdir(dirpath):
                dirname = os.path.basename(dirpath)
                # Vérifier que le nom commence bien par 'sub-'
                if dirname.startswith('sub-'):
                    sub_id = dirname.replace('sub-', '')
                    # Vérifier que l'ID du sujet est bien un nombre entier
                    # L'ensemble gère automatiquement les doublons, donc pas besoin de 'and sub_id not in ...'
                    if re.match(r'^\d+$', sub_id):
                        subject_ids_set.add(sub_id)
        
        self.subject_ids = sorted(list(subject_ids_set))  # Convertir en liste triée à la fin
    
    def refresh(self):
        """
        Forcer la réindexation de la base pour prendre en compte les changements.
        """
        folder_changed, old_hash = check_folder_changed(self.db_root, self.hash_file_path)
        
        if not folder_changed:
            print(f"Le dossier {self.db_root} n'a pas été modifié depuis la dernière indexation globale des sujets") # Message mis à jour
            # return # Optionnel: si on retourne ici, _index_subjects n'est pas appelé si le global n'a pas changé.
                     # Cependant, pour rafraîchir la liste des sujets, on continue.
        
        self._index_subjects()
        save_folder_hash(self.db_root, self.hash_file_path, {'global_db': True})
        self._subjects_cache.clear()  # Vider le cache des sujets car la liste des sujets peut avoir changé
    
    def get(self, sub_id, **kwargs):
        """
        Récupérer les fichiers d'un sujet qui correspondent aux critères donnés.
        """
        subject = self.get_subject(sub_id)
        return subject.get(**kwargs)
    
    def get_global(self, **kwargs):
        """
        Récupérer les fichiers globaux qui correspondent aux critères donnés.
        """
        import concurrent.futures
        
        if not self.subject_ids:
            self.refresh()
        if not self.subject_ids:
            raise ValueError("Aucun sujet trouvé dans la base de données BIDS.")
        
        results = []
        results_lock = threading.Lock()
        
        def search_subject(sub_id):
            try:
                subject_results = self.get(sub_id, **kwargs)
                with results_lock:
                    results.extend(subject_results)
            except Exception as e:
                print(f"Erreur lors de la recherche dans le sujet {sub_id}: {e}")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.subject_ids), os.cpu_count())) as executor:
            executor.map(search_subject, self.subject_ids)
        
        return results
    
    def get_subject(self, sub_id, force_index=False):
        """
        Retourner l'objet Subject associé à cet ID.
        """
        if force_index: # force_index ici déclenche Actidep.refresh()
            self.refresh()
        
        # self.subject_ids est mis à jour par self.refresh() si force_index était True
        if sub_id not in self.subject_ids:
            raise ValueError(f"Sujet {sub_id} non trouvé dans la base de données.")
        
        # Utiliser le cache des objets Subject
        if sub_id in self._subjects_cache:
            subject_instance = self._subjects_cache[sub_id]
            # S'assurer que l'objet Subject mis en cache est lui-même à jour
            # La méthode refresh de Subject est idempotente et vérifie son propre hash.
            subject_instance.refresh() 
            return subject_instance
        else:
            # Créer une nouvelle instance de Subject si non trouvée dans le cache
            subject_instance = Subject(sub_id, self.db_root)
            # La méthode __init__ de Subject gère sa propre logique d'indexation initiale
            # basée sur son propre fichier hash.
            self._subjects_cache[sub_id] = subject_instance
            return subject_instance
    
    def get_subjects(self):
        """
        Retourner la liste des ID de sujets.
        """
        return self.subject_ids
    
    def save_db(self):
        """
        Sauvegarder la base de données BIDS.
        """
        # Cette méthode est un placeholder, car les bases BIDS ne sont pas "sauvegardées" au sens traditionnel.
        print("La base de données BIDS a été sauvegardée.")
        return True


def test():
    """
    Fonction de test pour vérifier le fonctionnement de la classe Actidep et Subject.
    """
    import sys
    from actiDep.data.loader import Actidep, Subject

    test_path = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/mcm_tensors/sub-03011/dwi/sub-03011_desc-preproc_model-MCM_dwi.mcmx"
    # Create an instance of MyLoader
    print(f"Recherche dans {test_path}")
    ds = Actidep('/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    
    # Afficher les sujets trouvés
    print(f"Sujets trouvés: {ds.subject_ids}")
    
    # Vérifier si le chemin de test existe
    print(f"Le fichier existe: {os.path.exists(test_path)}")
    
    # Tester la récupération du fichier
    try:
        results = ds.get('03011', model='MCM', extension='mcmx', pipeline='mcm_tensors')
        print(f"Nombre de fichiers trouvés: {len(results)}")
        
        if len(results) > 0:
            test_get = results[0]
            print(f"Fichier trouvé: {test_get.path}")
            print(f"Égal au test_path: {test_get == test_path}")
            
            subject = Subject('03011', db_root='/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
            
            try:
                test_get = subject.get_unique(model='MCM', extension='mcmx', pipeline='mcm_tensors')
                print(f"get_unique a trouvé: {test_get.path}")
                print(f"Égal au test_path: {test_get == test_path}")
                
                #Build path
                test_entities = test_get.get_full_entities()
                print(f"Entités: {test_entities}")     
                test_build_path=subject.build_path(**test_entities)
                print(f"Chemin construit: {test_build_path}")
                print(f"Égal au test_path: {test_build_path == test_path}")
            except Exception as e:
                print(f"Erreur avec get_unique: {e}")
        else:
            print("Aucun fichier trouvé, vérifions le chemin de la base de données")
            # Vérifier si le dossier existe
            bids_path = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids'
            print(f"Le dossier de la base BIDS existe: {os.path.exists(bids_path)}")
            
            # Chercher tous les fichiers .mcmx dans la base
            print("Recherche de fichiers .mcmx...")
            mcmx_files = []
            for root, dirs, files in os.walk(bids_path):
                for file in files:
                    if file.endswith('.mcmx'):
                        mcmx_files.append(os.path.join(root, file))
                        
            print(f"Fichiers .mcmx trouvés: {len(mcmx_files)}")
            for f in mcmx_files[:5]:  # Afficher les 5 premiers fichiers
                print(f"  - {f}")
    except Exception as e:
        print(f"Erreur pendant l'exécution: {e}")


if __name__ == "__main__":
    # Pour exécuter ce script directement
    subject= Subject('01001', '/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids')
    print(f"Chargement du sujet {subject.bids_id} depuis {subject.db_root}")

    entities = {'suffix': 'dwi', 'desc': 'preproc', 'pipeline': 'anima_preproc', 'extension': 'nii.gz'}

    print(subject.get(**entities))