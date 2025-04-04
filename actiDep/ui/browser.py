import os
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from bids import BIDSLayout
from actiDep.set_config import set_config
from actiDep.data.loader import Subject, ActiDepFile
import pandas as pd
import json
import tempfile
import logging
import pathlib
import base64
import subprocess

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('actiDep.browser')

app = Flask(__name__)
CORS(app)

# Templates directory
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
app.template_folder = templates_dir

# Static files directory
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.static_folder = static_dir

# Créer les répertoires s'ils n'existent pas
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# Variables globales
config, tools = set_config()
db_root = '/home/ndecaux/Data/actidep_bids'
layout = None
# Cache pour stocker les sujets déjà chargés
subject_cache = {}

def init_layout():
    global layout
    if layout is None:
        layout = BIDSLayout(db_root, derivatives=True, validate=False)
    return layout

def get_subject(subject_id):
    """Récupère un objet Subject, avec mise en cache"""
    if subject_id not in subject_cache:
        subject_cache[subject_id] = Subject(subject_id, db_root=db_root)
    return subject_cache[subject_id]

@app.route('/')
def index():
    """Page d'accueil de l'interface web"""
    return render_template('index.html')

@app.route('/api/subjects')
def get_subjects():
    """Récupère la liste des sujets disponibles"""
    layout = init_layout()
    subjects = layout.get_subjects()
    return jsonify(subjects)

@app.route('/api/pipelines')
def get_pipelines():
    """Récupère la liste des pipelines disponibles"""
    layout = init_layout()
    pipelines = []
    
    # Parcourir les dossiers de dérivés
    derivatives_path = os.path.join(db_root, 'derivatives')
    if os.path.exists(derivatives_path):
        pipelines = [d for d in os.listdir(derivatives_path) 
                     if os.path.isdir(os.path.join(derivatives_path, d))]
    
    return jsonify(pipelines)

@app.route('/api/entities')
def get_entities():
    """Récupère la liste des entités disponibles"""
    subject_id = request.args.get('subject')
    
    # Si un sujet est spécifié, utiliser la méthode get_entity_values du sujet
    if subject_id:
        subject = get_subject(subject_id)
        entity_values = subject.get_entity_values()
        
        # Ajouter les entités personnalisées qui pourraient manquer
        custom_entities = ['pipeline', 'scope', 'model', 'desc', 'compartment', 'label']
        for entity in custom_entities:
            if entity not in entity_values:
                entity_values[entity] = []
                
        return jsonify({k: [] for k in entity_values.keys()})
    else:
        # Sinon, utiliser la méthode précédente
        layout = init_layout()
        entities = layout.get_entities()
        
        # Ajouter les entités personnalisées
        custom_entities = ['pipeline', 'scope', 'model', 'desc', 'compartment', 'label']
        
        all_entities = {**entities}
        for entity in custom_entities:
            if entity not in all_entities:
                all_entities[entity] = []
        
        return jsonify(all_entities)

@app.route('/api/entity_values')
def get_entity_values():
    """Récupère les valeurs possibles pour une entité donnée"""
    entity = request.args.get('entity')
    subject_id = request.args.get('subject')
    pipeline = request.args.get('pipeline')
    
    if not entity:
        return jsonify({'error': 'Entity parameter is required'}), 400
    
    # Si un sujet est spécifié, utiliser la méthode get_entity_values du sujet
    if subject_id:
        subject = get_subject(subject_id)
        entity_values = subject.get_entity_values()
        
        # Récupérer les valeurs pour l'entité demandée
        values = entity_values.get(entity, [])
        
        # Filtrer par pipeline si spécifié
        if pipeline and entity != 'pipeline':
            # Utiliser la méthode get du sujet pour filtrer directement par pipeline
            files = subject.get(pipeline=pipeline)
            filtered_values = set()
            
            for file in files:
                # Extraire la valeur de l'entité
                entities = file.get_full_entities()
                if entity in entities:
                    filtered_values.add(entities[entity])
            
            values = list(filtered_values)
            
        return jsonify(values)
    else:
        # Si aucun sujet n'est spécifié, utiliser la méthode précédente
        layout = init_layout()
        
        # Construire les filtres
        filters = {}
        
        # Pour les entités standard BIDS
        try:
            values = layout.get_values(entity, filters=filters)
        except ValueError:
            values = []
            
        # Pour les entités personnalisées comme pipeline ou model
        if entity == 'pipeline' and len(values) == 0:
            derivatives_path = os.path.join(db_root, 'derivatives')
            if os.path.exists(derivatives_path):
                values = [d for d in os.listdir(derivatives_path) 
                         if os.path.isdir(os.path.join(derivatives_path, d))]
        
        # Filtrer par pipeline si spécifié
        if pipeline and entity != 'pipeline':
            files = layout.get()
            filtered_values = set()
            
            for file in files:
                if f"derivatives/{pipeline}/" in file.path:
                    # Extraire la valeur de l'entité à partir du nom de fichier
                    entities = ActiDepFile(file).get_full_entities()
                    if entity in entities:
                        filtered_values.add(entities[entity])
            
            values = list(filtered_values)
        
        return jsonify(values)

@app.route('/api/search')
def search_files():
    """Recherche des fichiers selon les critères spécifiés"""
    layout = init_layout()
    
    # Récupérer tous les paramètres de requête
    query_params = request.args.to_dict()
    subject_id = query_params.get('subject')
    
    # Séparer les entités connues des entités personnalisées
    known_entities = layout.get_entities().keys()
    standard_params = {k: v for k, v in query_params.items() if k in known_entities}
    custom_params = {k: v for k, v in query_params.items() if k not in known_entities and k not in ['format']}
    
    # Format de sortie (json par défaut)
    output_format = query_params.get('format', 'json')
    
    # Si un sujet est spécifié, utiliser directement la méthode get du sujet
    if subject_id:
        subject = get_subject(subject_id)
        
        # Fusionner les paramètres pour la recherche
        search_params = {**standard_params, **custom_params}
        del search_params['subject']  # On le retire car déjà inclus dans l'objet Subject
        
        # Utiliser la méthode get améliorée du sujet
        results = subject.get(**search_params)
    else:
        # Ajouter le flag regex_search pour permettre des recherches plus flexibles
        standard_params['regex_search'] = True
        
        # Effectuer la recherche avec les entités standards
        results = layout.get(**standard_params)
        
        # Filtrer les résultats avec les entités personnalisées
        if custom_params:
            filtered_results = []
            for file in results:
                actidep_file = ActiDepFile(file)
                full_entities = actidep_file.get_full_entities()
                
                # Vérifier si les entités personnalisées correspondent
                match = True
                for key, value in custom_params.items():
                    if key == 'pipeline' and value:
                        if f"derivatives/{value}/" not in file.path:
                            match = False
                            break
                    elif key in full_entities and full_entities[key] != value:
                        match = False
                        break
                    elif key not in full_entities and value:
                        match = False
                        break
                
                if match:
                    filtered_results.append(actidep_file)
            
            results = filtered_results
        else:
            results = [ActiDepFile(file) for file in results]
    
    # Formater les résultats selon le format demandé
    if output_format == 'csv':
        # Créer un DataFrame pour export CSV
        data = []
        for file in results:
            entities = file.get_full_entities()
            entities['path'] = file.path
            data.append(entities)
        
        df = pd.DataFrame(data)
        
        # Créer un fichier temporaire pour le CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        return send_file(
            temp_file.name,
            mimetype='text/csv',
            as_attachment=True,
            download_name='search_results.csv'
        )
    else:
        # Format JSON par défaut
        results_json = []
        for file in results:
            result = {
                'path': file.path,
                'filename': file.filename,
                'entities': file.get_full_entities()
            }
            results_json.append(result)
        
        return jsonify(results_json)

@app.route('/api/subject_entities/<subject_id>')
def get_subject_entities(subject_id):
    """Récupère toutes les entités disponibles pour un sujet spécifique"""
    subject = get_subject(subject_id)
    entity_values = subject.get_entity_values()
    
    return jsonify(entity_values)

@app.route('/api/view/<path:file_path>')
def view_file(file_path):
    """Visualise un fichier en particulier"""
    full_path = os.path.join(db_root, file_path)
    
    # Vérifier que le fichier existe
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Obtenir les métadonnées du fichier
    file_info = {
        'path': full_path,
        'size': os.path.getsize(full_path),
        'modified': os.path.getmtime(full_path),
    }
    
    # Pour les images NIFTI, on pourrait ajouter un aperçu

    if full_path.endswith('.nii.gz') :
        # Launch ITK-SNAP in detached mode
        subprocess.Popen(['itksnap', '-g', full_path], 
                start_new_session=True,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL)
        
        # Return file info to the client
        return jsonify({**file_info, 'message': 'ITK-SNAP launched for this file'})

    # Pour les fichiers texte, on pourrait afficher le contenu
    
    # return jsonify(file_info)


def run_browser(host='localhost', port=5000, debug=True):
    """Démarre le serveur web"""
  
    # Lancer le serveur Flask
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_browser()