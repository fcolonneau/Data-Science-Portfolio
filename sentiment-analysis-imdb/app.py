import streamlit as st
from sentiment_analysis import SentimentAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("Analyseur de Sentiments pour Critiques de Films")

analyzer = SentimentAnalyzer()

# Champ de saisie pour une critique
user_input = st.text_area("Entrez votre critique de film :")
if st.button("Analyser"):
    if user_input:
        sentiment, probs = analyzer.predict_sentiment(user_input)
        st.write(f"**Sentiment prédit** : {sentiment}")
        
        # Visualisation des probabilités
        labels = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
        fig, ax = plt.subplots()
        sns.barplot(x=probs, y=labels)
        st.pyplot(fig)
    else:
        st.warning("Veuillez entrer une critique.")

# Option pour afficher les résultats du dataset IMDb
if st.button("Afficher les résultats IMDb"):
    if os.path.exists("results.csv"):
        results_df = pd.read_csv("results.csv")
        st.write("Résultats des 100 premières critiques IMDb :")
        st.dataframe(results_df[["text", "sentiment"]].head())
        st.image("visuals/sentiment_distribution.png", caption="Répartition des sentiments")
    else:
        st.warning("Veuillez d'abord exécuter sentiment_analysis.py pour générer les résultats.")