import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import os
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """Preprocess text for embeddings."""
    text = str(text).lower()
    if not text or text == "nan":
        return "no description available"
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return ' '.join(tokens)

@st.cache_resource
def _load_sentence_transformer():
    """Load SentenceTransformer model with CPU device."""
    logger.info("Loading SentenceTransformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def load_and_process_data():
    """Load dataset, preprocess, and compute embeddings for genre + description."""
    logger.info("Loading data...")
    try:
        df = pd.read_csv('top_movies.csv')
    except FileNotFoundError:
        logger.error("top_movies.csv not found.")
        raise FileNotFoundError("top_movies.csv not found. Please ensure the file exists in the project directory.")
    
    # Fill missing values
    df['genre'] = df['genre'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    
    # Preprocess description
    df['clean_description'] = df['description'].apply(preprocess_text)
    
    # Combine genre and clean_description for embeddings
    df['genre_description'] = df.apply(lambda row: f"{row['genre']} {row['clean_description']}".strip(), axis=1)
    
    if df.empty:
        logger.error("Dataset is empty.")
        raise ValueError("Dataset is empty.")
    
    logger.info("Preprocessing completed.")

    embeddings_file = 'data/embeddings.npy'
    os.makedirs('data', exist_ok=True)

    model = _load_sentence_transformer()
    if os.path.exists(embeddings_file):
        logger.info("Loading embeddings from file...")
        embeddings = np.load(embeddings_file)
        if embeddings.shape[0] != len(df):
            logger.warning("Embeddings file does not match dataset size. Recomputing embeddings...")
            embeddings = model.encode(df['genre_description'].tolist(), batch_size=32, show_progress_bar=True)
            np.save(embeddings_file, embeddings)
    else:
        logger.info("Computing embeddings...")
        embeddings = model.encode(df['genre_description'].tolist(), batch_size=32, show_progress_bar=True)
        np.save(embeddings_file, embeddings)
        logger.info(f"Embeddings saved to {embeddings_file}.")

    return df, embeddings, model