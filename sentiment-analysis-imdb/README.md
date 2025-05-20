# Analyseur de Sentiments pour Critiques de Films

Ce projet utilise un modèle BERT pré-entraîné pour analyser le sentiment (positif, négatif, neutre) de critiques de films, avec une intégration du dataset IMDb pour des analyses à grande échelle. Une interface Streamlit permet aux utilisateurs de saisir des critiques personnalisées et de visualiser les résultats, y compris une répartition des sentiments du dataset.

## Fonctionnalités
- **Analyse de sentiments** : Classification des critiques de films en cinq catégories ("Très négatif", "Négatif", "Neutre", "Positif", "Très positif") à l'aide du modèle BERT (`nlptown/bert-base-multilingual-uncased-sentiment`).
- **Traitement du dataset IMDb** : Analyse des critiques du dataset IMDb (limité à 100 échantillons pour la démonstration).
- **Interface utilisateur** : Application web interactive avec Streamlit pour saisir des critiques et afficher les résultats.
- **Visualisation** : Génération d'un histogramme montrant la répartition des sentiments dans le dataset IMDb.
- **Sortie des résultats** : Sauvegarde des analyses dans un fichier CSV (`results.csv`).

## Prérequis
- Python 3.11 (recommandé pour compatibilité optimale avec PyTorch et Streamlit).
- Un environnement virtuel est recommandé pour gérer les dépendances.
- Accès à internet pour télécharger le dataset IMDb et les modèles pré-entraînés.

## Installation
1. **Clonez le dépôt** :
   ```bash
   git clone https://github.com/fcolonneau/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```

2. **Créez un environnement virtuel** :
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # ou source venv/bin/activate  # Linux/Mac
   ```

3. **Installez les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   Contenu de `requirements.txt` :
   ```
   transformers==4.35.0
   torch==2.6.0
   pandas==2.0.3
   streamlit==1.38.0
   matplotlib==3.7.2
   seaborn==0.12.2
   datasets==2.20.0
   ```

4. **(Optionnel) Générez les résultats IMDb** :
   Exécutez le script pour analyser 100 critiques IMDb et générer un graphique :
   ```bash
   python sentiment_analysis.py
   ```

5. **Lancez l'application Streamlit** :
   ```bash
   python -m streamlit run app.py
   ```
   Ouvrez `http://localhost:8501` dans votre navigateur.

## Utilisation
- **Analyse manuelle** :
  - Saisissez une critique de film dans l'interface Streamlit.
  - Cliquez sur "Analyser" pour voir le sentiment prédit et un graphique des probabilités.
- **Résultats IMDb** :
  - Exécutez `sentiment-analysis-imdb.py` pour analyser 100 critiques IMDb et générer `results.csv` et `visuals/sentiment_distribution.png`.
  - Dans Streamlit, cliquez sur "Afficher les résultats IMDb" pour voir un tableau des 5 premières critiques et la répartition des sentiments.

## Structure du projet
```
sentiment-analysis-imdb/
├── app.py                 # Interface Streamlit
├── sentiment_analysis.py  # Logique d’analyse de sentiments
├── requirements.txt       # Dépendances
├── results.csv           # Résultats de l’analyse IMDb (généré)
├── visuals/              # Graphiques générés (ex. sentiment_distribution.png)
└── README.md             # Documentation
```

## Exemple de résultats
**Critique** : "Ce film est absolument fantastique, j'ai adoré chaque minute !"
**Sortie** : Positif

**Graphique** : Un histogramme montre la répartition des sentiments pour 100 critiques IMDb, avec des catégories comme "Positif" et "Négatif".
