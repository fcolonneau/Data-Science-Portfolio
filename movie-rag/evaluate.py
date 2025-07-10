# evaluate.py: Functions to evaluate the performance of the movie retrieval system.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging to capture evaluation events and errors.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_retrieval(df, embeddings, sentence_model, test_queries, k=5):
    """Evaluate retrieval performance using Precision@K and Recall@K.
    
    Args:
        df (pandas.DataFrame): DataFrame containing movie data.
        embeddings (numpy.ndarray): Precomputed embeddings for movie descriptions.
        sentence_model (SentenceTransformer): Model for encoding text into embeddings.
        test_queries (dict): Dictionary of test queries with relevant movie titles.
        k (int): Number of top results to evaluate (default: 5).
    
    Returns:
        list: List of dictionaries with precision and recall for each query.
        float: Average Precision@K across all queries.
        float: Average Recall@K across all queries.
    """
    from preprocessing import preprocess_text
    results = []
    
    # Iterate through each test query and its relevant movie IDs.
    for query, relevant_movie_ids in test_queries.items():
        # Preprocess query and compute its embedding.
        clean_query = preprocess_text(query)
        query_embedding = sentence_model.encode([clean_query], batch_size=1)
        
        # Compute cosine similarity between query and movie embeddings.
        desc_sim = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(desc_sim)[-k:][::-1]
        
        # Get indices of relevant movies from the DataFrame.
        relevant_indices = [df.index[df['movie_name'] == mid].tolist()[0] 
                           for mid in relevant_movie_ids if mid in df['movie_name'].values]
        
        # Calculate precision and recall for the query.
        retrieved = set(top_indices)
        relevant = set(relevant_indices)
        true_positives = len(retrieved.intersection(relevant))
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(relevant) if len(relevant) > 0 else 0
        
        # Store results for this query.
        results.append({
            'query': query,
            'precision@k': precision,
            'recall@k': recall
        })
    
    # Compute average precision and recall across all queries.
    avg_precision = np.mean([r['precision@k'] for r in results])
    avg_recall = np.mean([r['recall@k'] for r in results])
    
    # Log evaluation metrics.
    logger.info(f"Average Precision@{k}: {avg_precision:.3f}")
    logger.info(f"Average Recall@{k}: {avg_recall:.3f}")
    
    return results, avg_precision, avg_recall