# Apple Quality Prediction

Ce projet vise à prédire la qualité des pommes en utilisant une base de données de pommes. Le notebook est structuré comme suit :

## Introduction
Dans ce notebook, nous explorons une modélisation simple pour prédire la qualité des pommes en utilisant diverses techniques de traitement des données et de modélisation.

## Étapes du projet
1. **Nettoyage des données** : Nous commençons par nettoyer les données en éliminant les valeurs manquantes et en convertissant les variables catégorielles en variables numériques.
   
2. **Visualisation des données** : Ensuite, nous visualisons les données pour comprendre la distribution des caractéristiques et leur relation avec la qualité des pommes.

3. **Modèle MLPClassifier** : Nous construisons un modèle de prédiction de la qualité des pommes en utilisant MLPClassifier, un type de réseau de neurones à propagation avant.

## Description des colonnes
- **A_id** : L'ID de l'article.
- **Size** : La taille de l'article dans une unité de mesure donnée.
- **Weight** : Le poids de l'article dans une unité de mesure donnée.
- **Sweetness** : Le niveau de douceur de l'article sur une échelle allant de très sucré à très acide.
- **Crunchiness** : Le niveau de croustillant de l'article sur une échelle allant de très mou à très croquant.
- **Juiciness** : Le niveau de jutosité de l'article sur une échelle allant de très sec à très juteux.
- **Ripeness** : Le niveau de maturité de l'article sur une échelle allant de pas mûr à trop mûr.
- **Acidity** : Le niveau d'acidité de l'article sur une échelle allant de très faible à très élevé.
- **Quality** : La notation de qualité globale de l'article, soit "bonne", soit "mauvaise".

## Machine Learning
Nous avons utilisé MLPClassifier pour la modélisation. Voici quelques raisons justifiant son utilisation :
- Capacité à capturer des relations non linéaires.
- Adaptabilité à différents types de données.
- Flexibilité dans l'architecture du réseau.
- Adaptabilité à de grandes quantités de données.

## Résultats
Le modèle MLPClassifier a donné une précision moyenne de 91.64% sur cinq plis de validation croisée, ce qui indique qu'il est capable de prédire la qualité des pommes avec une performance satisfaisante.

---

Ce projet fournit une introduction à la prédiction de la qualité des pommes en utilisant MLPClassifier. Il offre également des opportunités pour des améliorations futures et des explorations plus approfondies dans le domaine.
