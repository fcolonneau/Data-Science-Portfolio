# Analyse de Sentiment des Tweets

Dans ce notebook, je présente une modélisation simple de l'analyse du sentiment d'une base de données de tweets liés à des entreprises spécifiques. Cette étude est structurée comme suit :

1. **Exploration initiale des données**
2. **Nettoyage des données**
3. **Modèle de Random Forest**
4. **Modèle de régression logistique**
5. **Modèle de SVM**

## Exploration initiale des données

Dans cette section, nous examinons les caractéristiques principales de notre jeu de données, telles que le nombre d'entités uniques, la distribution des sentiments, et la répartition des tweets par entité.

## Nettoyage des données

Nous effectuons différentes étapes de nettoyage des données, notamment la suppression des valeurs manquantes et des doublons, ainsi que le prétraitement du texte pour enlever les emojis, les URL et mettre le texte en minuscules.

## Tokenization et Lemmatisation

Nous expliquons brièvement les concepts de tokenization et de lemmatisation, et nous appliquons ces techniques pour prétraiter le texte des tweets, en utilisant la bibliothèque spaCy.

## Machine Learning

Dans cette partie, nous divisons nos données en ensembles d'entraînement et de test, puis nous utilisons trois modèles de classification différents pour prédire le sentiment des tweets : Random Forest, Régression Logistique et SVM.

## Analyse des performances des modèles

Nous évaluons les performances de chaque modèle en utilisant des mesures telles que la précision, le rappel et le score F1, et nous comparons leurs avantages et inconvénients pour la classification des sentiments des tweets.

Finalement, nous tirons des conclusions sur le meilleur modèle pour notre tâche d'analyse de sentiment des tweets.
