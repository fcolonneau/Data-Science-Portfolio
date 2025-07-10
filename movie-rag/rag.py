# rag.py: Retrieval-Augmented Generation (RAG) query processing for movie recommendations.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import logging

# Configure logging to capture query processing events and errors.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rag_query(query, df, embeddings, sentence_model, num_results=5, similarity_threshold=0.3):
    """Process a query using RAG and generate results based on similarity.
    
    Args:
        query (str): User's query about movies.
        df (pandas.DataFrame): DataFrame containing movie data.
        embeddings (numpy.ndarray): Precomputed embeddings for movie descriptions.
        sentence_model (SentenceTransformer): Model for encoding query into embeddings.
        num_results (int): Number of top results to return (default: 5).
        similarity_threshold (float): Minimum similarity score for results (default: 0.3).
    
    Returns:
        pandas.DataFrame: Top matching movies with similarity scores, or None if error occurs.
        str: Formatted response string or error message.
    """
    logger.info(f"Processing query: {query}")
    try:
        from preprocessing import preprocess_text
        # Validate query input.
        if not query.strip():
            logger.error("Empty query provided.")
            return None, "Please enter a valid query."
        
        # Preprocess query and compute its embedding.
        clean_query = preprocess_text(query)
        query_embedding = sentence_model.encode([clean_query], batch_size=1)
        
        # Compute cosine similarity between query and movie embeddings.
        desc_sim = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(desc_sim)[-num_results:][::-1]
        top_scores = desc_sim[top_indices]
        
        # Filter results by similarity threshold.
        results = df[['movie_name', 'genre', 'description']].iloc[top_indices].copy()
        results['similarity_score'] = top_scores
        results = results[results['similarity_score'] >= similarity_threshold]
        
        # Handle case where no results meet the threshold.
        if results.empty:
            logger.info("No results above similarity threshold.")
            return results, "No relevant movies found for your query."
        
        # Format response string with movie details and similarity scores.
        raw_response = "\n".join([f"- {row['movie_name']} ({row['genre']}): {row['description'][:100]}... (Score: {row['similarity_score']:.3f})" 
                                 for _, row in results.iterrows()])
        full_response = f"Detailed results:\n{raw_response}"
        
        return results, full_response
    except Exception as e:
        # Log and return error if query processing fails.
        logger.error(f"Error in rag_query: {e}")
        return None, f"Error processing query: {e}"