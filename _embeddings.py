from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import traceback
from copy import deepcopy
import pandas as pd
import requests
from bs4 import BeautifulSoup


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


def plot_sentiment(df):
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    colors = {'contractionary': 'red', 'neutral': 'gray', 'expansionary': 'green'}
    
    for i, (date, sentiment) in enumerate(zip(df_sorted.index, df_sorted['stance'])):
        plt.scatter(date, sentiment, c=colors[sentiment], alpha=0.7, s=10)
    
    for sentiment, color in colors.items():
        plt.scatter([], [], c=color, label=sentiment)
    
    plt.title('Treasury Sentiment Over Time')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_sentiment_score(df, threshold):
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    
    plt.plot(df_sorted.index, df_sorted['sentiment_score'], 'b.-', linewidth=1, alpha=0.7)
    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=-threshold, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.title('Treasury Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_stance_label(score, threshold):
    """Convert score to stance"""
    if score > threshold:
        return 'expansionary'
    elif score < -threshold:
        return 'contractionary'
    else:
        return 'neutral'

def analyze_treasury_statements(
    statements_df,
    labeled_examples,
    threshold,
    learn_threshold,   # NEW: stricter gate for self-training
    learn=True,
):
    if statements_df.empty:
        return pd.DataFrame()

    if labeled_examples is None:
        raise ValueError("Must provide labeled_examples with 'expansionary' and 'contractionary' keys")

    if 'expansionary' not in labeled_examples or 'contractionary' not in labeled_examples:
        raise ValueError("labeled_examples must have 'expansionary' and 'contractionary' keys")

    # Default: learn at same strictness as classification unless specified
    if learn_threshold is None:
        learn_threshold = threshold

    print(f"Analyzing {len(statements_df)} Treasury statements...")

    statements_df = statements_df.reset_index(drop=True)

    # Extract valid texts
    texts = []
    valid_indices = []
    for i, row in statements_df.iterrows():
        text = row.get('statement_text', '')
        if isinstance(text, str) and len(text.strip()) > 50:
            texts.append(text)
            valid_indices.append(i)

    if not texts:
        print("❌ No valid text content found")
        return pd.DataFrame()

    try:
        # 1) Embed docs ONCE
        print("Embed Docs ...")
        document_embeddings, model, _ = create_document_embeddings(texts)

        # 2) Build initial axis from current labeled examples
        print("Build Axis ...")
        axis = create_policy_axis(
            model,
            labeled_examples['expansionary'],
            labeled_examples['contractionary']
        )

        # 3) First scoring pass
        print("Get Scores ...")
        scores_v1 = calculate_sentiment_scores(document_embeddings, axis)

        # 4) First pass results + learning
        results = []
        learned_any = False

        # Keep a copy
        old_labeled_examples = deepcopy(labeled_examples)

        for i in range(min(len(scores_v1), len(valid_indices))):
            valid_idx = valid_indices[i]
            if valid_idx >= len(statements_df):
                continue

            row = statements_df.iloc[valid_idx]
            score = float(scores_v1[i])

            # Classification stance (for output)
            stance = get_stance_label(score, threshold)

            date = row.get('date')
            text = row.get('statement_text', '')

            # Learning stance (stricter gate)
            if learn:
                learn_stance = get_stance_label(score, learn_threshold)
                if learn_stance in ("expansionary", "contractionary") and not is_pure_auction_logistics_v2(text):

                    before = len(labeled_examples[learn_stance])

                    labeled_examples = integrate_new_labeled_examples(
                        labeled_examples=labeled_examples,
                        new_text=text,
                        stance=learn_stance,
                        model=model,
                    )

                    after = len(labeled_examples[learn_stance])
                    if after > before:
                        print(f"Added {learn_stance} example {date} (now {after})")
                        learned_any = True

            # store v1 temporarily (will be overwritten if we do pass 2)
            results.append({
                'date': date,
                'sentiment_score': score,
                'stance': stance
            })

        # Pop off old entries from labeled_examples
        for ex in old_labeled_examples['expansionary']:
            if ex in labeled_examples['expansionary']:
                labeled_examples['expansionary'].remove(ex)
        for ex in old_labeled_examples['contractionary']:
            if ex in labeled_examples['contractionary']:
                labeled_examples['contractionary'].remove(ex)

        # 5) If learning changed the example bank, rebuild axis ONCE and rescore
        if learn and learned_any:
            print("Update axis...")
            axis = create_policy_axis(
                model,
                labeled_examples['expansionary'],
                labeled_examples['contractionary']
            )
            scores_v2 = calculate_sentiment_scores(document_embeddings, axis)

            # overwrite scores/stances with v2 using same ordering
            for j in range(min(len(scores_v2), len(results))):
                score2 = float(scores_v2[j])
                results[j]['sentiment_score'] = score2
                results[j]['stance'] = get_stance_label(score2, threshold)

        results_df = pd.DataFrame(results)

        print(f"\n✓ Analysis complete (threshold={threshold}, learn_threshold={learn_threshold}):")
        print(f"  Expansionary: {(results_df['stance'] == 'expansionary').sum()}")
        print(f"  Neutral: {(results_df['stance'] == 'neutral').sum()}")
        print(f"  Contractionary: {(results_df['stance'] == 'contractionary').sum()}")

        if learn:
            print("\n✓ Updated labeled_examples (after pruning):")
            print(f"  expansionary: {len(labeled_examples.get('expansionary', []))}")
            print(f"  contractionary: {len(labeled_examples.get('contractionary', []))}")

        return results_df, labeled_examples

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return pd.DataFrame(), {}

