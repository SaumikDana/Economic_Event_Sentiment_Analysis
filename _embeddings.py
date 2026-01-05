from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler


def calculate_sentiment_scores(document_embeddings, axis):
    """Calculate sentiment scores by projecting onto policy axis"""
        
    raw_scores = np.dot(document_embeddings, axis)
    
    if len(raw_scores) > 1 and np.std(raw_scores) > 1e-10:
        sentiment_scores = StandardScaler().fit_transform(raw_scores.reshape(-1, 1)).flatten()
    else:
        sentiment_scores = raw_scores
    
    return sentiment_scores


def create_document_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Create sentence transformer embeddings"""
    
    print(f"Processing {len(texts)} documents with {model_name}...")
    
    model = SentenceTransformer(model_name)
    document_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Created embeddings with shape: {document_embeddings.shape}")
    
    return document_embeddings, model, texts


def create_policy_axis(model, expansionary_examples, contractionary_examples):
    """Create fiscal policy axis from labeled example statements
    
    Args:
        model: Sentence transformer model
        expansionary_examples: List of actual Treasury statements that are expansionary
        contractionary_examples: List of actual Treasury statements that are contractionary
        
    Returns:
        axis: Direction vector representing expansionary→contractionary in embedding space
    """
    
    if len(expansionary_examples) < 3 or len(contractionary_examples) < 3:
        raise ValueError(f"Need at least 3 examples per class. Got {len(expansionary_examples)} expansionary and {len(contractionary_examples)} contractionary")
    
    print(f"Creating policy axis from {len(expansionary_examples)} expansionary and {len(contractionary_examples)} contractionary examples...")
    
    # Embed the example statements
    expansionary_embeddings = model.encode(expansionary_examples, convert_to_numpy=True)
    contractionary_embeddings = model.encode(contractionary_examples, convert_to_numpy=True)
    
    # Get centroids
    expansionary_centroid = np.mean(expansionary_embeddings, axis=0)
    contractionary_centroid = np.mean(contractionary_embeddings, axis=0)
    
    # Create axis
    axis = expansionary_centroid - contractionary_centroid
    
    # Normalize
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
        print(f"✓ Policy axis created (separation: {norm:.3f})")
        
        # Check quality
        exp_projection = np.dot(expansionary_embeddings, axis).mean()
        con_projection = np.dot(contractionary_embeddings, axis).mean()
        separation = abs(exp_projection - con_projection)
        print(f"✓ Class separation: {separation:.3f} (higher = better)")
    else:
        raise ValueError("Policy axis has zero norm - examples may be identical")
    
    return axis