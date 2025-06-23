import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rag_query(query, df, embeddings, sentence_model, num_results=5, similarity_threshold=0.3):
    """Process a query using RAG and generate an introductory hook with DistilGPT2."""
    logger.info(f"Processing query: {query}")
    try:
        from preprocessing import preprocess_text
        if not query.strip():
            logger.error("Empty query provided.")
            return None, "Please enter a valid query."
        
        clean_query = preprocess_text(query)
        query_embedding = sentence_model.encode([clean_query], batch_size=1)

        # Retrieve top results
        desc_sim = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(desc_sim)[-num_results:][::-1]
        top_scores = desc_sim[top_indices]
        results = df[['movie_name', 'genre', 'description']].iloc[top_indices].copy()
        results['similarity_score'] = top_scores

        # Filter results by similarity threshold
        results = results[results['similarity_score'] >= similarity_threshold]

        if results.empty:
            logger.info("No results above similarity threshold.")
            return results, "No relevant movies found for your query."

        # Combine raw results and hook
        raw_response = "\n".join([f"- {row['movie_name']} ({row['genre']}): {row['description'][:100]}... (Score: {row['similarity_score']:.3f})" for _, row in results.iterrows()])
        full_response = f"Detailed results:\n{raw_response}"

        return results, full_response
    except Exception as e:
        logger.error(f"Error in rag_query: {e}")
        return None, f"Error processing query: {e}"