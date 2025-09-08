import os
import json
import pathlib
import nibabel as nib
import SimpleITK as sitk
import re
from os.path import join as opj
from actiDep.data.io import copy2nii, move2nii, parse_filename

_SENTINEL = object()  # Défini au niveau du module

class LayoutProxy:
    """
    Une classe proxy qui simule l'interface d'un layout PyBIDS.
    """
    def __init__(self, root, parent_obj=None):
        self.root = root
        self._parent_obj = parent_obj
    
    def get_subjects(self):
        """Retourner la liste des sujets depuis l'objet parent."""
        if hasattr(self._parent_obj, 'get_subjects'):
            return self._parent_obj.get_subjects()
        elif hasattr(self._parent_obj, 'subject_ids'):
            return self._parent_obj.subject_ids
        else:
            return []
    
    def get(self, **kwargs):
        """Déléguer la recherche à l'objet parent."""
        if hasattr(self._parent_obj, 'get_global'):
            return self._parent_obj.get_global(**kwargs)
        elif hasattr(self._parent_obj, 'get'):
            return self._parent_obj.get(**kwargs)
        else:
            return []

class ActiDepFile:
    """
    Représentation minimale d'un fichier BIDS (allégée).
    """
    def __init__(self, path, row=None):
        self.path = path
        self.filename = os.path.basename(path)
        if row is not None:
            # Utiliser directement les informations du DataFrame en filtrant valeurs non pertinentes
            def _is_nan(v):
                try:
                    return v != v  # NaN != NaN
                except Exception:
                    return False
            self.entities = {k: v for k, v in row.items()
                             if k not in ['path'] and v is not None and not _is_nan(v)}
        else:
            # Fallback: parsing basique
            self.entities = parse_filename(self.filename)
            if '.' in self.filename:
                self.entities['extension'] = self.filename[self.filename.find('.'):]
            match_sub = re.search(r'/sub-([A-Za-z0-9]+)/', path)
            if match_sub:
                self.entities['subject'] = match_sub.group(1)
            match_ses = re.search(r'/ses-([A-Za-z0-9]+)/', path)
            if match_ses:
                self.entities['session'] = match_ses.group(1)
        # Drapeaux dérivés
        self.entities['derivative'] = 'derivatives' in path
        if 'suffix' not in self.entities:
            stem = self.filename.split('.')[0]
            last = stem.split('_')[-1]
            if '-' not in last:
                self.entities['suffix'] = last

    def __getattr__(self, item):
        """Permet l'accès direct (file.model, file.extension, etc.) comme dans l'ancienne implémentation.
        Déclenché uniquement si l'attribut n'existe pas déjà sur l'objet.
        """
        if 'entities' in self.__dict__ and item in self.entities:
            return self.entities[item]
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{item}'")

    @property
    def extension(self):  # compat explicite si code externe attend une propriété
        return self.entities.get('extension')

    @property
    def subject(self):
        """Accès direct à l'identifiant sujet (pour compatibilité d'usage)."""
        return self.entities.get('subject')

    @property
    def session(self):
        """Accès direct à l'identifiant session si présent."""
        return self.entities.get('session')

    def get_full_entities(self):
        return dict(self.entities)

    def get_entities(self):
        return {k: v for k, v in self.entities.items() if k not in ['path', 'derivative']}
    def copy(self, dest):
        return copy2nii(self.path, dest)

    def move(self, dest):
        return move2nii(self.path, dest)

    def __eq__(self, other):
        return self.path == (other.path if isinstance(other, ActiDepFile) else other)

    def __hash__(self):
        return hash(self.path)

    def __str__(self):
        return self.path

class Subject:
    """
    Sujet léger basé uniquement sur le DataFrame global de Actidep.
    """
    def __init__(self, sub_id, db_root="/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids", layout=None, parent_actidep=None):
        self.sub_id = sub_id
        self.db_root = db_root
        self.bids_id = f"sub-{sub_id}"
        self.layout = layout or LayoutProxy(db_root, self)
        self.parent_actidep = parent_actidep
        self.dicom_folder = opj(self.db_root, self.bids_id, 'sourcedata', f'{self.bids_id}_dicoms')

    def get(self, **kwargs):
        # Gestion du scope avant filtrage
        scope = kwargs.pop('scope', None)
        if self.parent_actidep is None:
            self.parent_actidep = Actidep(self.db_root)
        df = self.parent_actidep.build_dataframe()
        if df.empty:
            return []
        sub_df = df[df['subject'] == self.sub_id]
        if sub_df.empty:
            return []
        # Filtrage scope=raw => uniquement non-derivatives
        if scope == 'raw':
            sub_df = sub_df[sub_df['derivative'] != True]
            if sub_df.empty:
                return []
        if 'subject' in kwargs:
            kwargs.pop('subject', None)
        filtered = self.parent_actidep._apply_filters(sub_df, kwargs)
        if filtered.empty:
            return []
        return [ActiDepFile(row['path'], row) for _, row in filtered.iterrows()]

    def get_entity_values(self):
        if self.parent_actidep is None:
            return {}
        df = self.parent_actidep.build_dataframe()
        sub_df = df[df['subject'] == self.sub_id]
        out = {}
        for col in sub_df.columns:
            if col in ['path', 'subject']:
                continue
            vals = [v for v in sub_df[col].dropna().unique().tolist() if v != '']
            if vals:
                out[col] = sorted(vals)
        return out

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

    def build_path(self, suffix, datatype=None, pipeline=None, session=None,
                   extension='.nii.gz', original_name=None, is_dir=False, **entities):
        """Construire un chemin BIDS (raw ou derivative) avec:
        - entités triées alphabétiquement
        - extension déduite de original_name si extension reste par défaut
        - exclusion des valeurs None / NaN / 'nan'
        - suppression des clés contenant un underscore
        - support de is_dir (pas d'extension, création dossier)
        """
        # 1. Nettoyage des entités
        cleaned = {}
        for k, v in list(entities.items()):
            if v is None:
                continue
            try:
                if v != v:  # NaN
                    continue
            except Exception:
                pass
            if isinstance(v, str) and v.strip().lower() == 'nan':
                continue
            cleaned[k] = v

        # 2. Champs réservés
        reserved = {'subject', 'derivative', 'datatype', 'pipeline', 'extension', 'suffix', 'session', 'is_dir'}

        # Propagation
        if 'datatype' in cleaned and datatype is None:
            datatype = cleaned['datatype']
        if 'pipeline' in cleaned and pipeline is None:
            pipeline = cleaned['pipeline']
        if 'session' in cleaned and session is None:
            session = cleaned['session']
        if 'suffix' in cleaned and suffix is None:
            suffix = cleaned['suffix']
        if 'is_dir' in cleaned:
            try:
                is_dir = bool(cleaned['is_dir'])
            except Exception:
                pass

        # 3. Filtrage entités pour le nom
        entity_items = {k: v for k, v in cleaned.items() if k not in reserved and '_' not in k}

        # 4. Extension
        if is_dir:
            resolved_ext = ''
        else:
            if extension is None:
                resolved_ext = ''
            else:
                if original_name and (extension in ('.nii.gz', '.nii', '.gz') or extension in ('', None)):
                    parts = original_name.split('.')
                    if len(parts) > 2 and parts[-2].lower() == 'nii' and parts[-1].lower() == 'gz':
                        resolved_ext = '.nii.gz'
                    elif len(parts) > 1:
                        resolved_ext = '.' + parts[-1]
                    else:
                        resolved_ext = extension if (extension.startswith('.') if extension else False) else (f'.{extension}' if extension else '')
                else:
                    resolved_ext = extension if extension.startswith('.') else f'.{extension}' if extension else ''

        # 5. Datatype
        datatype = datatype or suffix
        if not datatype:
            raise ValueError("datatype ou suffix requis pour construire le chemin")

        # 6. Base
        if pipeline:
            base = opj(self.db_root, 'derivatives', pipeline, self.bids_id)
        else:
            base = opj(self.db_root, self.bids_id)
        if session:
            base = opj(base, f"ses-{session}")
        base = opj(base, datatype)
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)

        # 7. Nom
        name_parts = [self.bids_id]
        if session:
            name_parts.append(f"ses-{session}")
        for k in sorted(entity_items.keys()):
            name_parts.append(f"{k}-{entity_items[k]}")
        name = '_'.join(name_parts) + f"_{suffix}{resolved_ext}"
        full_path = opj(base, name)
        if is_dir:
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        return full_path

    def write_object(self, obj, suffix, **kwargs):
        path = self.build_path(suffix, **kwargs)
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if isinstance(obj, sitk.Image):
            sitk.WriteImage(obj, path)
        elif isinstance(obj, nib.Nifti1Image):
            nib.save(obj, path)
        elif isinstance(obj, dict):
            with open(path, 'w') as f:
                json.dump(obj, f)
        elif isinstance(obj, str) and os.path.exists(obj):
            copy2nii(obj, path)
        else:
            raise ValueError("Type non supporté pour write_object.")
        return path

class Actidep:
    """
    Gestion simplifiée de la base BIDS via un DataFrame unique.
    """
    def __init__(self, db_root='/home/ndecaux/Data/actidep_bids'):
        self.db_root = db_root
        self.layout = LayoutProxy(db_root, self)
        self._df = None
        self._subjects_cache = {}
        self.subject_ids = []  # sera peuplé par build_dataframe()
        # Pré-construction pour exposer immédiatement les sujets
        try:
            self.build_dataframe()
        except Exception:
            # On ignore pour laisser une initialisation paresseuse si ça échoue
            pass

    def build_dataframe(self, force=False):
        if self._df is not None and not force:
            return self._df
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Installer pandas pour utiliser Actidep.") from e

        rows = []

        def add_file(filepath, pipeline=None, derivative=False, session=None, datatype=None, subject=None):
            fname = os.path.basename(filepath)
            if not fname.startswith('sub-'):
                return
            # Extension
            if '.' in fname:
                dot = fname.find('.')
                ext = fname[dot:]
                stem = fname[:dot]
            else:
                ext, stem = '', fname
            parts = stem.split('_')
            ent = {}
            subj = subject
            ses = session
            for p in parts:
                if p.startswith('sub-'):
                    subj = p[4:]
                elif p.startswith('ses-'):
                    ses = ses or p[4:]
                elif '-' in p:
                    k, v = p.split('-', 1)
                    ent[k] = v
            suffix_candidate = parts[-1]
            suffix = suffix_candidate if '-' not in suffix_candidate else ent.get('suffix', suffix_candidate)
            row = {
                'path': filepath,
                'subject': subj,
                'session': ses,
                'datatype': datatype,
                'pipeline': pipeline,
                'derivative': derivative,
                'extension': ext,
                'suffix': suffix
            }
            for k, v in ent.items():
                if k not in row:
                    row[k] = v
            rows.append(row)

        if not os.path.isdir(self.db_root):
            import pandas as pd
            self._df = pd.DataFrame()
            self.subject_ids = []
            return self._df

        # Raw
        for root, dirs, files in os.walk(self.db_root):
            if '/derivatives/' in root or root.endswith('/derivatives'):
                continue
            # Déterminer subject / session / datatype
            parts = root.replace(self.db_root, '').strip('/').split('/')
            subject = None
            session = None
            datatype = None
            for part in parts:
                if part.startswith('sub-'):
                    subject = part[4:]
                elif part.startswith('ses-'):
                    session = part[4:]
                elif subject and datatype is None:
                    # Premier dossier après subject(/session)
                    if part not in ['', 'sourcedata']:
                        datatype = part
            for f in files:
                if 'sourcedata' in root:
                    continue
                add_file(os.path.join(root, f), derivative=False, subject=subject,
                         session=session, datatype=datatype)

        # Derivatives
        derivatives_root = opj(self.db_root, 'derivatives')
        if os.path.isdir(derivatives_root):
            for root, dirs, files in os.walk(derivatives_root):
                rel = root.replace(derivatives_root, '').strip('/')
                if rel == '':
                    continue
                parts = rel.split('/')
                pipeline = parts[0] if parts else None
                subject = None
                session = None
                datatype = None
                for part in parts[1:]:
                    if part.startswith('sub-'):
                        subject = part[4:]
                    elif part.startswith('ses-'):
                        session = part[4:]
                    elif subject and datatype is None:
                        if part not in ['', 'sourcedata']:
                            datatype = part
                for f in files:
                    if 'sourcedata' in root:
                        continue
                    add_file(os.path.join(root, f), pipeline=pipeline, derivative=True,
                             subject=subject, session=session, datatype=datatype)

        import pandas as pd
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.dropna(subset=['subject']).drop_duplicates(subset=['path']).reset_index(drop=True)
            self.subject_ids = sorted(df['subject'].unique().tolist())
        else:
            self.subject_ids = []
        self._df = df
        return self._df

    def _apply_filters(self, df, kwargs):
        if df.empty or not kwargs:
            return df
        import numpy as np
        mask = np.ones(len(df), dtype=bool)
        for k, v in kwargs.items():
            # Normalisation spécifique pour l'extension: accepter sans point initial
            if k == 'extension' and isinstance(v, str):
                neg = v.startswith('!')
                core = v[1:] if neg else v
                if not core.startswith('.'):
                    core = '.' + core
                v = ('!' if neg else '') + core
            if k not in df.columns:
                if isinstance(v, str) and v.startswith('!'):
                    continue
                return df.iloc[0:0]
            col = df[k].astype(str)
            if isinstance(v, str) and v.startswith('!'):
                target = v[1:]
                mask &= (col != target)
            elif v is None:
                mask &= (~df[k].notna())
            else:
                mask &= (col == str(v))
            if not mask.any():
                return df.iloc[0:0]
        return df[mask]

    def get(self, sub_id, **kwargs):
        # Intercepter scope
        scope = kwargs.pop('scope', None)
        df = self.build_dataframe()
        if df.empty:
            return []
        sdf = df[df['subject'] == sub_id]
        if sdf.empty:
            return []
        if scope == 'raw':
            sdf = sdf[sdf['derivative'] != True]
            if sdf.empty:
                return []
        if 'subject' in kwargs:
            kwargs.pop('subject')
        sdf = self._apply_filters(sdf, kwargs)
        if sdf.empty:
            return []
        subj = self.get_subject(sub_id)
        return [ActiDepFile(r['path'], r) for _, r in sdf.iterrows()]

    def get_global(self, **kwargs):
        # Support scope=raw pour ignorer derivatives
        scope = kwargs.pop('scope', None)
        df = self.build_dataframe()
        if df.empty:
            return []
        if scope == 'raw':
            df = df[df['derivative'] != True]
            if df.empty:
                return []
        df = self._apply_filters(df, kwargs)
        if df.empty:
            return []
        # cache sujets
        cache = {}
        out = []
        for _, r in df.iterrows():
            sid = r['subject']
            if sid not in cache:
                cache[sid] = self.get_subject(sid)
            out.append(ActiDepFile(r['path'], r))
        return out

    def get_subject(self, sub_id):
        if sub_id in self._subjects_cache:
            return self._subjects_cache[sub_id]
        self.build_dataframe()
        if self.subject_ids and sub_id not in self.subject_ids:
            raise ValueError(f"Sujet {sub_id} introuvable.")
        subj = Subject(sub_id, self.db_root, layout=self.layout, parent_actidep=self)
        self._subjects_cache[sub_id] = subj
        return subj

    def to_dataframe(self):
        return self.build_dataframe()

    def refresh(self):
        self.build_dataframe(force=True)
        self._subjects_cache.clear()
        return True

# Test simplifié
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
    test()