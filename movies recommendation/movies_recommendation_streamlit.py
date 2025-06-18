import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Télécharger les ressources NLTK (exécuter une fois)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger le dataset
df = pd.read_csv('top_movies.csv')

# Gérer les valeurs manquantes dans 'genre'
df['genre'] = df['genre'].fillna('').astype(str)

# Fonction de prétraitement du texte
def preprocess_text(text):
    """
    Prétraiter le texte : convertir en minuscules, tokeniser, supprimer les stop-words,
    lemmatiser, et retourner une chaîne propre.
    """
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return ' '.join(tokens)

# Appliquer le prétraitement à la colonne 'description'
df['clean_description'] = df['description'].apply(preprocess_text)

# Calculer les embeddings avec Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['clean_description'].tolist())
cosine_sim_embeddings = cosine_similarity(embeddings)

# Fonction pour calculer la similarité des genres (Jaccard)
def genre_similarity(genres1, genres2):
    """
    Calculer la similarité Jaccard entre deux ensembles de genres.
    Gérer les cas où les genres sont vides ou non-valides.
    """
    if not genres1 or not genres2:
        return 0.0
    set1 = set(genres1.split(', '))
    set2 = set(genres2.split(', '))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Fonction de recommandation
def recommend_movies(movie_title, num_recommendations=5, desc_weight=0.7, genre_weight=0.3):
    """
    Recommander des films similaires à un titre donné, en combinant la similarité des
    descriptions (embeddings) et des genres (Jaccard).
    """
    if movie_title not in df['movie_name'].values:
        return pd.DataFrame(), f"Le film '{movie_title}' n'est pas dans le dataset."
    
    movie_idx = df[df['movie_name'] == movie_title].index[0]
    desc_sim = cosine_sim_embeddings[movie_idx]
    target_genres = df['genre'].iloc[movie_idx]
    genre_sim = [genre_similarity(target_genres, df['genre'].iloc[i]) for i in range(len(df))]
    combined_sim = desc_weight * desc_sim + genre_weight * np.array(genre_sim)
    sim_scores = list(enumerate(combined_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [idx for idx, _ in sim_scores]
    recommendations = df[['movie_name', 'genre', 'description']].iloc[movie_indices].copy()
    recommendations['similarity_score'] = [score for _, score in sim_scores]
    return recommendations, None

# Interface Streamlit
st.title("Système de Recommandation de Films")
st.write("Choisissez un film et obtenez des recommandations basées sur les descriptions et genres.")

# Sélection du film
movie_title = st.selectbox("Choisissez un film", df['movie_name'].tolist())

# Paramètres ajustables
num_recommendations = st.slider("Nombre de recommandations", 1, 10, 5)
desc_weight = st.slider("Poids de la similarité des descriptions", 0.0, 1.0, 0.7)
genre_weight = 1.0 - desc_weight  # Assurer que les poids somment à 1
st.write(f"Poids de la similarité des genres : {genre_weight:.2f}")

# Bouton pour lancer les recommandations
if st.button("Recommander"):
    recommendations, error = recommend_movies(movie_title, num_recommendations, desc_weight, genre_weight)
    if error:
        st.error(error)
    else:
        st.write(f"Recommandations pour '{movie_title}':")
        st.dataframe(recommendations)