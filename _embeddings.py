from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from sklearn.metrics.pairwise import cosine_similarity


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


def normalize_text_for_dedupe(text: str) -> str:
    """Cheap normalization to catch exact/near-exact duplicates."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text


def prune_semantic_redundancy(
    texts,
    model,
    similarity_threshold=0.92,
    max_examples=None,
):
    """
    Removes semantically redundant texts using embedding cosine similarity.
    Keeps the first occurrence of each "cluster".
    """
    if not texts:
        return []

    # Fast exact-ish dedupe first
    seen = set()
    unique = []
    for t in texts:
        nt = normalize_text_for_dedupe(t)
        if nt not in seen and len(nt) > 20:
            seen.add(nt)
            unique.append(t)

    if len(unique) <= 1:
        return unique

    embs = model.encode(unique, convert_to_numpy=True, show_progress_bar=False)
    kept_texts = []
    kept_embs = []

    for t, e in zip(unique, embs):
        if not kept_embs:
            kept_texts.append(t)
            kept_embs.append(e)
            continue

        sims = cosine_similarity([e], np.vstack(kept_embs))[0]
        if np.max(sims) < similarity_threshold:
            kept_texts.append(t)
            kept_embs.append(e)

        if max_examples is not None and len(kept_texts) >= max_examples:
            break

    return kept_texts


def integrate_new_labeled_examples(
    labeled_examples,
    new_text,
    stance,
    model,
    similarity_threshold=0.92,
    max_per_class=200,
):
    """
    Adds a new statement into labeled_examples[stance] (expansionary/contractionary),
    then prunes redundancy.
    """
    if stance not in ("expansionary", "contractionary"):
        return labeled_examples  # ignore neutral / unknown

    if not isinstance(new_text, str) or len(new_text.strip()) < 50:
        return labeled_examples

    labeled_examples.setdefault("expansionary", [])
    labeled_examples.setdefault("contractionary", [])

    # Append then prune
    labeled_examples[stance].append(new_text)

    labeled_examples[stance] = prune_semantic_redundancy(
        labeled_examples[stance],
        model=model,
        similarity_threshold=similarity_threshold,
        max_examples=max_per_class,
    )

    return labeled_examples


import requests
from bs4 import BeautifulSoup

def extract_clean_text(soup):
    """Extract clean text from Treasury page"""
    
    category = ""
    category_elem = soup.select_one('.news-category')
    if category_elem:
        category = category_elem.get_text().strip()
    
    title = ""
    title_elem = soup.select_one('h2.uswds-page-title span')
    if title_elem:
        title = title_elem.get_text().strip()
    
    date = ""
    date_elem = soup.select_one('.field--name-field-news-publication-date time')
    if date_elem:
        date = date_elem.get_text().strip()
    
    body_text = ""
    body_elem = soup.select_one('.field--name-field-news-body')
    if body_elem:
        paragraphs = []
        for p in body_elem.find_all('p'):
            text = p.get_text().strip()
            if text:
                paragraphs.append(text)
        body_text = '\n\n'.join(paragraphs)
    
    full_text_parts = []
    if category:
        full_text_parts.append(category)
    if title:
        full_text_parts.append(title)
    if date:
        full_text_parts.append(date)
    if body_text:
        full_text_parts.append(body_text)
    
    return '\n'.join(full_text_parts)


def extract_text(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.content, 'html.parser')
    return extract_clean_text(soup)


def scrape_text(prefix, num):
    """Scrape Treasury press release"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    base_url = "https://home.treasury.gov/news/press-releases/"
    alt_base_url = "https://home.treasury.gov/news/press-release/"

    url = f"{base_url}{prefix}{num:04d}"
    text_content = extract_text(url, headers)

    if not text_content:
        url = f"{alt_base_url}{prefix}{num:04d}"
        text_content = extract_text(url, headers)

    if not text_content and len(str(abs(num))) == 3:
        url = f"{base_url}{prefix}{num:03d}"
        text_content = extract_text(url, headers)
        if not text_content:
            url = f"{alt_base_url}{prefix}{num:03d}"
            text_content = extract_text(url, headers)

    if not text_content:
        return None, None

    release_date = re.search(r'(\b\w+\s+\d{1,2},\s+\d{4})(?=\s*\(Archived Content\))', text_content)
    if not release_date:
        early_text = text_content[:1000]
        release_date = re.search(r'(\b\w+\s+\d{1,2},\s+\d{4})', early_text)

    release_date = release_date.group(1) if release_date else None
    if release_date is None:
        return None, None

    return text_content, release_date



AUCTION_KEYWORDS = {
    "auction", "reopening", "refund", "refunding", "issuance",
    "treasury will auction", "competitive bids", "noncompetitive",
    "cusip", "settlement date", "issue date", "maturity",
    "bill", "note", "bond", "tips",
    "offering amount", "offering sizes",
    "announces the auction", "auction results",
    "when-issued", "wi",
    "primary dealers", "federal reserve bank of new york",
}

FISCAL_STANCE_KEYWORDS = {
    # spending / stimulus
    "spending", "outlays", "investment", "infrastructure",
    "support households", "support businesses", "stimulus",
    "relief", "aid", "assistance",

    # taxes
    "tax", "taxes", "tax relief", "tax cut", "tax increase",
    "credits", "rebate",

    # deficit / debt framing
    "deficit", "deficits", "debt", "fiscal",
    "budget", "borrowing", "sustainability",
    "fiscal space", "austerity", "consolidation",

    # macro intent
    "economic growth", "employment", "jobs", "demand",
    "inflation", "countercyclical", "recovery",
}

def is_pure_auction_logistics_v2(text: str) -> bool:
    """
    Return True if text is operational debt-auction content
    with no fiscal stance language.
    """
    if not isinstance(text, str):
        return True

    t = text.lower()

    # Count keyword hits
    auction_hits = sum(1 for kw in AUCTION_KEYWORDS if kw in t)
    fiscal_hits = sum(1 for kw in FISCAL_STANCE_KEYWORDS if kw in t)

    # Heuristics:
    # - At least some auction language
    # - Zero fiscal stance language
    # - Mostly operational length / structure
    return (
        auction_hits >= 2 and
        fiscal_hits == 0
    )
