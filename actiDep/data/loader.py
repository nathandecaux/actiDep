from bids import BIDSLayout
import bids
from os.path import join as opj
import os
import SimpleITK as sitk
import shutil
import pathlib
from ..utils.tools import del_key, upt_dict, create_pipeline_description
from .io import parse_filename, convertNRRDToNifti, copy2nii, move2nii, copy_list, copy_from_dict
import ants
import json
from pprint import pprint
import nibabel as nib

class ActiDepFile(bids.layout.BIDSFile):
    def __init__(self, bids_file, subject=None):
        vars(self).update(vars(bids_file))
        self.bids_file = bids_file
        self.scope = 'derivatives' if 'derivatives' in self.path else 'raw'
        self.pipeline = self.path.split(
            'derivatives/')[1].split('/')[0] if self.scope == 'derivatives' else None
        self.extra_entities = parse_filename(self.filename)
        self.extra_entities.pop('sub', None)
        self.subject_obj = subject

        # Set entities as attributes
        for k, v in self.get_full_entities().items():
            setattr(self, k, v)

    def copy(self, dest):
        """Copy the file to a new location, converting if necessary."""
        return copy2nii(self.path, dest)

    def move(self, dest):
        """Move the file to a new location, converting if necessary."""
        return move2nii(self.path, dest)

    def get_full_entities(self, **kwargs):
        """
        Another wrapper around a BIDSLayout method.
        In addition to the 'official' entities, it also returns the custom entities, scopes and pipelines.
        """

        entities = self.bids_file.get_entities(**kwargs)
        entities['scope'] = self.scope
        entities['pipeline'] = self.pipeline
        entities.update(self.extra_entities)
        return entities

    def get_subject(self):
        """
        Return the Subject object associated with this file.
        """
        if self.subject_obj is not None:
            return self.subject_obj
        else:
            return Subject(self.entities['subject'])


class Subject:
    def __init__(self, sub_id, db_root='/home/ndecaux/Code/Data/actidep_bids'):
        self.db_root = db_root
        self.sub_id = sub_id
        self.bids_id = f'sub-{sub_id}'
        self.layout = BIDSLayout(
            self.db_root, derivatives=True, validate=False)
        self.dicom_folder = opj(
            self.db_root, self.bids_id, 'sourcedata', f'{self.bids_id}_dicoms')

    def get(self, **kwargs):
        """
        Recover files that match the given criteria from the BIDS dataset.
        eg: get(model='DTI',metric='FA') should return files with _model-DTI_ and _metric-FA_ in their name.
        Basically a wrapper around the BIDSLayout.get() method that also handles custom entities.
        """
        # Get all kwargs that are known entities
        self.layout = BIDSLayout(
            self.db_root, derivatives=True, validate=False
        )
        official_entities = list(self.layout.get_entities().keys())
        other_entries = ['return_type', 'target', 'scope',
                         'regex_search', 'absolute_paths', 'invalid_filters']
    
        # Séparer les entités connues et les entités personnalisées
        known_entities = {k: kwargs[k] for k in kwargs.keys() 
                         if k in official_entities+other_entries}
        custom_entities = {k: v for k, v in kwargs.items() 
                          if k not in known_entities.keys()}
    
        # Préparer les paramètres pour la requête BIDS
        query_params = {'subject': self.sub_id, 'regex_search': True}
        
        # Traiter les entités "None" et les négations (!) pour les entités connues
        for entity, value in list(known_entities.items()):
            if value is None:
                # Si la valeur est None, on n'ajoute pas ce critère à la requête
                custom_entities.update({entity:known_entities.pop(entity)})

            elif isinstance(value, str) and '!' in value:
                # Si on veut exclure une valeur, on construit une regex d'exclusion
                excluded_value = value.replace('!', '')
                known_entities[entity] = f'^((?!{excluded_value}).)*$'
        
        # Ajouter les entités connues restantes à la requête
        query_params.update(known_entities)
        
        # Récupérer la liste initiale de fichiers
        sub_list = self.layout.get(**query_params)
        
        # Filtrer par pipeline si spécifié
        if 'pipeline' in custom_entities:
            pipeline = custom_entities.pop('pipeline')
            # Filter files that don't contains derivative/<pipeline> in their path
            sub_list = [f for f in sub_list if f.path.find(
                f'derivatives/{pipeline}/') != -1]
        
        # Filtrer par entités personnalisées
        files = []
        for f in sub_list:
            entities = parse_filename(f.filename)
            try:
                matched = True
                for k, v in custom_entities.items():
                    if v is None:
                        # Si la valeur est None, on vérifie que la clé n'existe pas dans les entités
                        if k in entities:
                            matched = False
                            break
                    elif isinstance(v, str) and '!' in v:
                        # Si la valeur commence par !, on vérifie que la clé n'existe pas avec une valeur spécifique
                        if entities.get(k) == v[1:]:
                            matched = False
                            break
                    else:
                        # Comportement habituel: vérification que la clé existe avec la valeur attendue
                        if entities.get(k) != v:
                            matched = False
                            break
                
                if matched:
                    actidep_file = ActiDepFile(f, self)
                    files.append(actidep_file)
            except KeyError:
                # Si une clé n'existe pas et qu'on cherche une valeur spécifique (pas None)
                continue
        
        return files

    def get_unique(self, **kwargs):
        """
        Return the unique file that matches the given criteria.
        """
        files = self.get(**kwargs)
        if len(files) == 0:
            raise ValueError(f"No file found for {kwargs}")
        if len(files) > 1:
            raise ValueError(f"Multiple files found for {kwargs}")
        return files[0]

    def build_path(self, suffix, pipeline=None, scope=None, original_name=None, is_dir=False, **kwargs):
        """
        Construire le chemin d'un fichier dans le layout à partir d'entités et d'un pipeline donnés.

        Args:
            suffix (str): Suffixe du fichier (dwi, T1w, etc.)
            pipeline (str): Nom de la pipeline (anima_preproc, etc.)
            original_name (str): Nom du fichier original
            **kwargs: Entités supplémentaires à ajouter au nom du fichier
        """
        entities = {'suffix': suffix}

        for k, v in kwargs.items():
            entities[k] = v

        root_dir = self.layout.root
        parent_dir = entities.pop(
            'datatype') if 'datatype' in entities else entities['suffix']
        if pipeline is None:
            target_dir = opj(root_dir, self.bids_id, parent_dir)
        else:
            if pipeline not in self.get_pipelines():
                create_pipeline_description(pipeline, self.layout)

            target_dir = opj(root_dir, 'derivatives',
                             pipeline, self.bids_id, parent_dir)

        fileradix = f'{self.bids_id}'

        if original_name is not None and 'extension' not in entities:
            # entities['desc'] = original_name.split('_')[-1].split('.')[0]
            extension = '.'+'.'.join(original_name.split('.')[1:])
            if extension == '.nrrd':
                extension = '.nii.gz'

        else:
            
            is_dir = entities.get('is_dir') or is_dir
            
            if 'extension' in entities:
                extension = '.' + entities['extension'].lstrip('.')
            else:
                
                extension = '' if is_dir else '.nii.gz'
                # Display a warning
                print(
                    f"Warning: No extension provided for {entities}, assuming {extension}")

        remaining_entities = {k: v for k, v in entities.items() if k not in [
            'suffix', 'subject', 'extension']}

        for k, v in sorted(remaining_entities.items()):
            fileradix += f'_{k}-{v}'
        path = opj(target_dir, fileradix+"_"+entities['suffix']+extension)

        return path

    def get_pipelines(self):
        """
        Return the list of pipelines that have been run on this subject.
        """
        pipelines = set()
        for f in self.layout.get(subject=self.sub_id):
            if 'derivatives' in f.path:
                pipeline = f.path.split('derivatives/')[1].split('/')[0]
                pipelines.add(pipeline)
        return list(pipelines)

    def write_object(self, obj, suffix, pipeline=None, scope=None, original_name=None, **kwargs):
        """
        Write an object to a file in the layout.
        """
        path = self.build_path(suffix, pipeline, scope,
                               original_name, **kwargs)
        
        if isinstance(obj, sitk.Image):
            sitk.WriteImage(obj, path)

        elif isinstance(obj, ants.ANTsImage):
            obj.to_filename(path)

        elif isinstance(obj, dict):
            with open(path, 'w') as f:
                json.dump(obj, f)
            
        elif isinstance(obj, nib.Nifti1Image):
            nib.save(obj, path)

        elif isinstance(obj, str):
            copy2nii(obj, path)
        else:
            raise ValueError(
                f"Object type {type(obj)} not supported for writing.")

        return path


class Actidep:
    def __init__(self, db_root='/home/ndecaux/Code/Data/actidep_bids'):
        __abstract__ = True
        self.db_root = db_root
        self.layout = BIDSLayout(self.db_root, derivatives=True)
        self.subjects = self.layout.get_subjects()

    def get(self, sub_id, **kwargs):
        subject = Subject(sub_id)
        return subject.get(**kwargs)

    def get_subjects(self):
        return self.subjects


if __name__ == "__main__":
    subject = Subject('03011')
    sub = subject.get(model='DTI', metric='FA')[0]
    print(sub.filename)
    # Print the full path

    example = subject.get(suffix='dwi', extension='bvec', scope='raw')

    print(example)
