from imports import *


def calculate_sentiment_scores(document_embeddings, axis):
    """Calculate sentiment scores with debugging"""
        
    # Project documents onto fiscal axis
    raw_scores = np.dot(document_embeddings, axis)
    
    # Only standardize if we have variation
    if len(raw_scores) > 1 and np.std(raw_scores) > 1e-10:
        sentiment_scores = StandardScaler().fit_transform(raw_scores.reshape(-1, 1)).flatten()
    else:
        sentiment_scores = raw_scores
    
    return sentiment_scores


def create_document_embeddings(texts):
    """Create TF-IDF embeddings with improved preprocessing"""

    print(f"Processing {len(texts)} documents...")
            
    # Adjusted TF-IDF parameters
    vectorizer = TfidfVectorizer(
        max_features=min(1000, len(texts) * 20),
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,
        max_df=0.8,  # More restrictive on common terms
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',
        sublinear_tf=True  # Apply log scaling to tf
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        document_embeddings = tfidf_matrix.toarray()
                
        return document_embeddings, vectorizer, texts
        
    except ValueError as e:
        print(f"TF-IDF failed with error: {e}")
        raise

