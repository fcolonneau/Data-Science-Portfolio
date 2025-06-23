import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_retrieval(df, embeddings, sentence_model, test_queries, k=5):
    """Evaluate retrieval performance using Precision@K and Recall@K."""
    from preprocessing import preprocess_text
    results = []
    
    for query, relevant_movie_ids in test_queries.items():
        clean_query = preprocess_text(query)
        query_embedding = sentence_model.encode([clean_query], batch_size=1)
        desc_sim = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(desc_sim)[-k:][::-1]
        
        # Convert movie IDs to indices (assuming movie_name is unique)
        relevant_indices = [df.index[df['movie_name'] == mid].tolist()[0] for mid in relevant_movie_ids if mid in df['movie_name'].values]
        
        # Calculate Precision@K and Recall@K
        retrieved = set(top_indices)
        relevant = set(relevant_indices)
        true_positives = len(retrieved.intersection(relevant))
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(relevant) if len(relevant) > 0 else 0
        
        results.append({
            'query': query,
            'precision@k': precision,
            'recall@k': recall
        })
    
    # Compute average metrics
    avg_precision = np.mean([r['precision@k'] for r in results])
    avg_recall = np.mean([r['recall@k'] for r in results])
    
    logger.info(f"Average Precisionprincipale Precision@{k}: {avg_precision:.3f}")
    logger.info(f"Average Recall@{k}: {avg_recall:.3f}")
    
    return results, avg_precision, avg_recall