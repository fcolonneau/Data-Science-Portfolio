import streamlit as st
import os
import logging
from transformers import pipeline
from preprocessing import load_and_process_data
from rag import rag_query
from evaluate import evaluate_retrieval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Suppress transformers warnings
logging.getLogger('transformers').setLevel(logging.ERROR)


# Streamlit app
st.title("Movie RAG Assistant")
st.write("Ask a question about movies (e.g., 'Which movies are about the mafia?')")
st.warning("Please enter your question in English.")
st.info("Loading data and models, please wait...")

# Load data and models
try:
    df, embeddings, sentence_model = load_and_process_data()
    st.success("Data and models loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# User input
query = st.text_input("Your question (in English)")
num_results = st.slider("Number of results", 1, 10, 5)

# Run query
if st.button("Search"):
    if query:
        results, response = rag_query(query, df, embeddings, sentence_model, num_results)
        if results is not None:
            st.write(response)
            if not results.empty:
                st.dataframe(results)
                with st.expander("View full descriptions"):
                    for _, row in results.iterrows():
                        st.write(f"**{row['movie_name']}** ({row['genre']}): {row['description']}")
            else:
                st.warning("No relevant results found.")
        else:
            st.error(response)
    else:
        st.error("Please enter a question.")

# Evaluation section
if st.button("Run Evaluation"):
    test_queries = {
    "mafia movies": [
        "The Transporter Refueled", "Bugsy", "Scarface", "The Godfather", "The Godfather Part II",
        "The Godfather Part III", "Goodfellas", "Donnie Brasco", "The Family", "The Departed",
        "One Hundred Steps"
    ],
    "horror movies": [
        "Scream", "Final Cut", "The Shining", "Halloween", "A Nightmare on Elm Street",
        "The Exorcist", "Scary Movie 2", "Scary Movie", "Tales of Halloween", "Grave Encounters", "Grave Encounters 2", "The Hills Run Red",
        "Behind the Mask", "Final Cut"
    ],
    "comedy movies": [
        "Hot Shots!", "Micmacs", "Airplane!", "The Naked Gun", "Superbad", "Anchorman: The Legend of Ron Burgundy", "Bad Tri", "Top Five",
        "Everybody Wants Some!!", "Evolution"
    ],
    "action movies": [
        "Die Hard", "Mad Max: Fury Road", "The Dark Knight", "John Wick", "Speed",
        "Mission: Impossible – Fallout", "Hot Shots!", "The Transporter Refueled", "2048: Nowhere to Run", "Last Action Hero", "Kung Pow"
    ],
    "heist movies": [
        "Heat", "The Italian Job", "Ocean's Eleven", "Snatch", "Reservoir Dogs",  "Lying and Stealing", 
        "Inside Man", "The Getaway", "The Score", "Los Bandoleros", "Thunderbolt and Lightfoot"
    ],
    "romance movies": [
        "Titanic", "The Notebook", "Amélie", "La La Land", "Before Sunrise", "The Little Death", "Malcolm & Marie", "Kiki: Love to Love", "Closer ",
        "Pride and Prejudice", "Emma", "Hotel Chevalier", "Lisbela and the Prisoner", "Romeo and Juliet"
    ],
    "war movies": [
        "Saving Private Ryan", "Apocalypse Now", "Dunkirk", "Platoon", "Full Metal Jacket", "Jarhead", "Escape to Victory", "Tropic Thunder", "Patton ",
        "1917", "Beasts of No Nation", "Wonder Woman", "Sobibor "
    ]
}
    results, avg_precision, avg_recall = evaluate_retrieval(df, embeddings, sentence_model, test_queries, k=num_results)
    st.write(f"**Evaluation Results**")
    st.write(f"Average Precision@{num_results}: {avg_precision:.3f}")
    st.write(f"Average Recall@{num_results}: {avg_recall:.3f}")
    st.dataframe(results)