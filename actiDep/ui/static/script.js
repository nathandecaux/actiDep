// Variables globales
    let entities = {};
    let entityValues = {}; // Stockage des valeurs d'entités par sujet
    let currentSubject = '';
    let currentPipeline = '';
    
    // Initialisation au chargement de la page
    document.addEventListener('DOMContentLoaded', () => {
        // Charger les sujets
        fetch('/api/subjects')
            .then(response => response.json())
            .then(data => {
                const subjectSelect = document.getElementById('subject');
                data.forEach(subject => {
                    const option = document.createElement('option');
                    option.value = subject;
                    option.textContent = subject;
                    subjectSelect.appendChild(option);
                });
            });
        
        // Charger les pipelines
        fetch('/api/pipelines')
            .then(response => response.json())
            .then(data => {
                const pipelineSelect = document.getElementById('pipeline');
                data.forEach(pipeline => {
                    const option = document.createElement('option');
                    option.value = pipeline;
                    option.textContent = pipeline;
                    pipelineSelect.appendChild(option);
                });
            });
        
        // Charger les entités disponibles
        fetch('/api/entities')
            .then(response => response.json())
            .then(data => {
                entities = data;
            });
        
        // Gérer les événements
        document.getElementById('searchForm').addEventListener('submit', handleSearch);
        document.getElementById('addFilterBtn').addEventListener('click', addFilter);
        document.getElementById('subject').addEventListener('change', handleSubjectChange);
        document.getElementById('pipeline').addEventListener('change', handlePipelineChange);
        
        // Nouveaux gestionnaires d'événements pour la recherche libre
        document.getElementById('freeSearchBtn').addEventListener('click', handleFreeSearch);
        document.getElementById('freeSearchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleFreeSearch();
                e.preventDefault();
            }
        });
    });
    
    // Nouvelle fonction pour gérer la recherche libre
    function handleFreeSearch() {
        // On récupère le texte à rechercher
        const searchText = document.getElementById('freeSearchInput').value.trim();
        
        // Construire les paramètres de recherche
        const params = new URLSearchParams();
        
        // Ajouter le sujet et le pipeline seulement s'ils ne sont pas vides
        const subject = document.getElementById('subject').value;
        const pipeline = document.getElementById('pipeline').value;
        
        if (subject) {
            params.append('subject', subject);
        }
        
        if (pipeline) {
            params.append('pipeline', pipeline);
        }
        
        // Ajouter les filtres dynamiques
        const filterRows = document.querySelectorAll('.filter-row');
        filterRows.forEach(row => {
            const entitySelect = row.querySelector('.entity-select');
            const valueSelect = row.querySelector('.value-select');
            
            if (entitySelect.value && valueSelect.value) {
                params.append(entitySelect.value, valueSelect.value);
            }
        });
        
        // Ajouter le texte de recherche libre
        if (searchText) {
            params.append('search_text', searchText);
        }
        
        // Ajouter le format de sortie
        const format = document.getElementById('outputFormat').value;
        params.append('format', format);
        
        // Afficher un indicateur de chargement
        const resultsTable = document.getElementById('resultsTable');
        const resultsCount = document.getElementById('resultsCount');
        resultsCount.textContent = "Recherche en cours...";
        resultsTable.innerHTML = '<tr><td colspan="3" class="text-center">Chargement...</td></tr>';
        
        // Si format est CSV, télécharger directement
        if (format === 'csv') {
            window.location.href = `/api/search?${params.toString()}`;
            return;
        }
        
        // Faire la requête avec les filtres et la recherche libre
        fetch(`/api/search?${params.toString()}`)
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                resultsCount.textContent = "Erreur lors de la recherche";
                resultsTable.innerHTML = `<tr><td colspan="3" class="text-danger">Erreur: ${error.message}</td></tr>`;
            });
    }
    
    function handleSubjectChange(event) {
        currentSubject = event.target.value;
        
        if (currentSubject) {
            // Charger toutes les entités et leurs valeurs pour ce sujet
            fetch(`/api/subject_entities/${currentSubject}`)
                .then(response => response.json())
                .then(data => {
                    entityValues[currentSubject] = data;
                    updateFilterEntities();
                });
        }
        
        // Mettre à jour les valeurs des filtres existants
        updateFilterValues();
    }
    
    function updateFilterEntities() {
        // Mettre à jour les listes déroulantes des entités dans les filtres existants
        const filterRows = document.querySelectorAll('.filter-row');
        filterRows.forEach(row => {
            const entitySelect = row.querySelector('.entity-select');
            if (entitySelect) {
                const currentValue = entitySelect.value;
                
                // Vider le sélecteur d'entités
                entitySelect.innerHTML = '<option value="">Sélectionner une entité</option>';
                
                // Le remplir avec les entités disponibles pour le sujet courant
                if (currentSubject && entityValues[currentSubject]) {
                    for (const entity in entityValues[currentSubject]) {
                        const option = document.createElement('option');
                        option.value = entity;
                        option.textContent = entity;
                        option.selected = (entity === currentValue);
                        entitySelect.appendChild(option);
                    }
                } else {
                    // Si pas de sujet sélectionné, utiliser les entités globales
                    for (const entity in entities) {
                        const option = document.createElement('option');
                        option.value = entity;
                        option.textContent = entity;
                        option.selected = (entity === currentValue);
                        entitySelect.appendChild(option);
                    }
                }
            }
        });
    }
    
    function handlePipelineChange(event) {
        currentPipeline = event.target.value;
        // Mettre à jour les valeurs des filtres existants
        updateFilterValues();
    }
    
    function updateFilterValues() {
        // Pour chaque filtre existant, mettre à jour ses options de valeur
        const filterRows = document.querySelectorAll('.filter-row');
        filterRows.forEach(row => {
            const entitySelect = row.querySelector('.entity-select');
            const valueSelect = row.querySelector('.value-select');
            
            if (entitySelect && valueSelect && entitySelect.value) {
                const entityName = entitySelect.value;
                updateEntityValues(entityName, valueSelect);
            }
        });
    }
    
    function updateEntityValues(entityName, valueSelect) {
        // Vider les options actuelles
        valueSelect.innerHTML = '<option value="">Toutes les valeurs</option>';
        
        // Si nous avons déjà chargé les valeurs pour ce sujet et cette entité
        if (currentSubject && entityValues[currentSubject] && entityValues[currentSubject][entityName]) {
            const values = entityValues[currentSubject][entityName];
            
            // Filtrer par pipeline si spécifié
            if (currentPipeline && entityName !== 'pipeline') {
                // On utilisera quand même l'API pour le filtrage par pipeline
                const url = `/api/entity_values?entity=${entityName}&subject=${currentSubject}&pipeline=${currentPipeline}`;
                fetch(url)
                    .then(response => response.json())
                    .then(values => {
                        populateValueSelect(values, valueSelect);
                    });
            } else {
                // Utiliser les valeurs en cache
                populateValueSelect(values, valueSelect);
            }
        } else {
            // Récupérer les nouvelles valeurs pour l'entité sélectionnée via l'API
            const url = `/api/entity_values?entity=${entityName}&subject=${currentSubject}&pipeline=${currentPipeline}`;
            
            fetch(url)
                .then(response => response.json())
                .then(values => {
                    populateValueSelect(values, valueSelect);
                });
        }
    }
    
    function populateValueSelect(values, valueSelect) {
        // Ajouter les options au sélecteur
        values.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            valueSelect.appendChild(option);
        });
    }
    
    function addFilter() {
        const dynamicFilters = document.getElementById('dynamicFilters');
        const filterRow = document.createElement('div');
        filterRow.className = 'filter-row';
        
        // Créer le sélecteur d'entité
        const entitySelect = document.createElement('select');
        entitySelect.className = 'form-select mb-2 entity-select';
        
        // Option par défaut
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Sélectionner une entité';
        entitySelect.appendChild(defaultOption);
        
        // Ajouter les entités disponibles (selon le sujet courant si sélectionné)
        if (currentSubject && entityValues[currentSubject]) {
            for (const entity in entityValues[currentSubject]) {
                const option = document.createElement('option');
                option.value = entity;
                option.textContent = entity;
                entitySelect.appendChild(option);
            }
        } else {
            // Si pas de sujet sélectionné, utiliser les entités globales
            for (const entity in entities) {
                const option = document.createElement('option');
                option.value = entity;
                option.textContent = entity;
                entitySelect.appendChild(option);
            }
        }
        
        // Créer le sélecteur de valeur
        const valueSelect = document.createElement('select');
        valueSelect.className = 'form-select value-select';
        valueSelect.innerHTML = '<option value="">Toutes les valeurs</option>';
        
        // Ajouter un bouton pour supprimer le filtre
        const removeButton = document.createElement('span');
        removeButton.className = 'remove-filter badge bg-danger';
        removeButton.textContent = '×';
        removeButton.addEventListener('click', () => {
            dynamicFilters.removeChild(filterRow);
        });
        
        // Réagir au changement d'entité pour charger les valeurs correspondantes
        entitySelect.addEventListener('change', (e) => {
            const entityName = e.target.value;
            if (entityName) {
                updateEntityValues(entityName, valueSelect);
            } else {
                // Réinitialiser le sélecteur de valeur
                valueSelect.innerHTML = '<option value="">Toutes les valeurs</option>';
            }
        });
        
        // Assembler le tout
        filterRow.appendChild(removeButton);
        filterRow.appendChild(entitySelect);
        filterRow.appendChild(valueSelect);
        dynamicFilters.appendChild(filterRow);
    }
    
    function handleSearch(e) {
        e.preventDefault();
        
        // Construire les paramètres de recherche
        const params = new URLSearchParams();
        
        // Ajouter le sujet et le pipeline seulement s'ils ne sont pas vides
        const subject = document.getElementById('subject').value;
        const pipeline = document.getElementById('pipeline').value;
        
        if (subject) {
            params.append('subject', subject);
        }
        
        if (pipeline) {
            params.append('pipeline', pipeline);
        }
        
        // Ajouter les filtres dynamiques
        const filterRows = document.querySelectorAll('.filter-row');
        filterRows.forEach(row => {
            const entitySelect = row.querySelector('.entity-select');
            const valueSelect = row.querySelector('.value-select');
            
            if (entitySelect.value && valueSelect.value) {
                params.append(entitySelect.value, valueSelect.value);
            }
        });
        
        // Ajouter le texte de recherche libre s'il existe
        const searchText = document.getElementById('freeSearchInput').value.trim();
        if (searchText) {
            params.append('search_text', searchText);
        }
        
        // Ajouter le format de sortie
        const format = document.getElementById('outputFormat').value;
        params.append('format', format);
        
        // Si format est CSV, télécharger directement
        if (format === 'csv') {
            window.location.href = `/api/search?${params.toString()}`;
            return;
        }
        
        // Afficher un indicateur de chargement
        const resultsTable = document.getElementById('resultsTable');
        const resultsCount = document.getElementById('resultsCount');
        resultsCount.textContent = "Recherche en cours...";
        resultsTable.innerHTML = '<tr><td colspan="3" class="text-center">Chargement...</td></tr>';
        
        // Sinon afficher les résultats dans le tableau
        fetch(`/api/search?${params.toString()}`)
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                resultsCount.textContent = "Erreur lors de la recherche";
                resultsTable.innerHTML = `<tr><td colspan="3" class="text-danger">Erreur: ${error.message}</td></tr>`;
            });
    }
    
    function displayResults(results) {
        const resultsTable = document.getElementById('resultsTable');
        const resultsCount = document.getElementById('resultsCount');
        
        // Afficher le nombre de résultats
        resultsCount.textContent = `${results.length} fichier(s) trouvé(s)`;
        
        // Vider le tableau
        resultsTable.innerHTML = '';
        
        if (results.length === 0) {
            resultsTable.innerHTML = '<tr><td colspan="3" class="text-center">Aucun résultat trouvé</td></tr>';
            return;
        }
        
        // Remplir le tableau avec les résultats
        results.forEach(file => {
            const row = document.createElement('tr');
            
            // Colonne fichier avec style pour permettre le retour à la ligne
            const fileCell = document.createElement('td');
            fileCell.style.wordWrap = 'break-word';
            fileCell.style.maxWidth = '250px';
            fileCell.textContent = file.filename;
            row.appendChild(fileCell);
            
            // Colonne entités
            const entitiesCell = document.createElement('td');
            for (const [key, value] of Object.entries(file.entities)) {
                if (value) {
                    const badge = document.createElement('span');
                    badge.className = 'entity-badge badge bg-info text-dark';
                    badge.textContent = `${key}: ${value}`;
                    entitiesCell.appendChild(badge);
                }
            }
            row.appendChild(entitiesCell);
            
            // Colonne actions
            const actionsCell = document.createElement('td');
            
            // Bouton pour visualiser le fichier
            const viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-sm btn-outline-primary me-2';
            viewBtn.textContent = 'Ouvrir';
            viewBtn.addEventListener('click', () => {
                // Extraire le chemin relatif à partir du chemin complet
                const relativePath = file.path.split('/actidep_bids/')[1];
                window.open(`/api/view/${relativePath}`, '_blank');
            });
            actionsCell.appendChild(viewBtn);

            // Bouton pour ouvrir le dossier du fichier
            const folderBtn = document.createElement('button');
            folderBtn.className = 'btn btn-sm btn-outline-secondary me-2';
            folderBtn.textContent = '📁';
            folderBtn.addEventListener('click', () => {
                // Extraire le chemin relatif à partir du chemin complet
                const relativePath = file.path.split('/actidep_bids/')[1];
                window.open(`/api/folder/${relativePath}`, '_blank');
            });
            actionsCell.appendChild(folderBtn);
            
            // Nouveau bouton pour copier le chemin complet
            const copyBtn = document.createElement('button');
            copyBtn.className = 'btn btn-sm btn-outline-info';
            copyBtn.textContent = '📋';
            copyBtn.title = 'Copier le chemin complet';
            copyBtn.addEventListener('click', () => {
                copyToClipboard(file.path);
                // Ajouter un retour visuel temporaire
                const originalText = copyBtn.textContent;
                copyBtn.textContent = '✓';
                copyBtn.classList.add('btn-success');
                copyBtn.classList.remove('btn-outline-info');
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.classList.remove('btn-success');
                    copyBtn.classList.add('btn-outline-info');
                }, 1000);
            });
            actionsCell.appendChild(copyBtn);
            
            row.appendChild(actionsCell);
            
            resultsTable.appendChild(row);
        });
    }

    // Fonction utilitaire pour copier du texte dans le presse-papier
    function copyToClipboard(text) {
        // Créer un élément temporaire
        const textArea = document.createElement('textarea');
        textArea.value = text;
        
        // Le rendre invisible
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            // Copier le texte
            const successful = document.execCommand('copy');
            if (!successful) {
                console.error('Impossible de copier le texte');
            }
        } catch (err) {
            console.error('Erreur lors de la copie:', err);
        }
        
        // Nettoyer
        document.body.removeChild(textArea);
    }
