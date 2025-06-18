import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import logging
import warnings
import tensorflow as tf

# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignorer les logs TensorFlow
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Télécharger les ressources NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Fonction de prétraitement du texte
def preprocess_text(text):
    """Prétraiter le texte pour les embeddings."""
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return ' '.join(tokens)

# Charger et préparer les données
@st.cache_data(persist=True)
def load_and_process_data():
    """Charger le dataset, prétraiter, calculer les embeddings et sauvegarder."""
    logger.info("Chargement des données...")
    df = pd.read_csv('top_movies.csv')
    df['genre'] = df['genre'].fillna('').astype(str)
    df['clean_description'] = df['description'].apply(preprocess_text)
    logger.info("Prétraitement terminé.")

    embeddings_file = 'embeddings.npy'
    try:
        if os.path.exists(embeddings_file):
            logger.info("Chargement des embeddings depuis le fichier de cache.")
            embeddings = np.load(embeddings_file)
        else:
            logger.info("Calcul des embeddings...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(df['clean_description'].tolist(), batch_size=32, show_progress_bar=True)
            np.save(embeddings_file, embeddings)
            logger.info(f"Embeddings sauvegardés dans {embeddings_file}.")
    except Exception as e:
        logger.error(f"Erreur lors du calcul ou de la sauvegarde : {e}")
        raise e

    return df, embeddings

# Fonction RAG
def rag_query(query, df, embeddings, num_results=5):
    """Répondre à une question en utilisant RAG (basé sur les descriptions)."""
    logger.info(f"Traitement de la question : {query}")
    clean_query = preprocess_text(query)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([clean_query], batch_size=1)

    desc_sim = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(desc_sim)[-num_results:][::-1]
    results = df[['movie_name', 'genre', 'description']].iloc[top_indices].copy()
    results['similarity_score'] = desc_sim[top_indices]

    response = f"Results for '{query}':\n"
    for _, row in results.iterrows():
        response += f"- {row['movie_name']} ({row['genre']}): {row['description'][:100]}... (Score: {row['similarity_score']:.3f})\n"
    
    return results, response

# Interface Streamlit
st.title("Movie RAG Assistant")
st.write("Ask a question about movies (e.g., 'Which movies are about the mafia?')")
st.warning("Please enter your question in English.")
st.info("Loading data, please wait...")

# Charger les données
try:
    df, embeddings = load_and_process_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Entrée utilisateur
query = st.text_input("Your question (in English)")
num_results = st.slider("Number of results", 1, 10, 5)

# Bouton pour lancer la requête
if st.button("Search"):
    if query:
        results, response = rag_query(query, df, embeddings, num_results)
        st.write(response)
        st.dataframe(results)
    else:
        st.error("Please enter a question.")