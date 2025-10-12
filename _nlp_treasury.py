from imports import *
from _embeddings import create_document_embeddings, calculate_sentiment_scores


def plot_sentiment(df):

    # Sort dataframe by date index
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    # Create scatter plot
    plt.figure(figsize=(14, 9))
    colors = {'contractionary': 'red', 'neutral': 'gray', 'expansionary': 'green'}
    
    # Plot each point in chronological order
    for i, (date, sentiment) in enumerate(zip(df_sorted.index, df_sorted['stance'])):
        plt.scatter(date, sentiment, c=colors[sentiment], alpha=0.7, s=10)
    
    # Add legend manually
    for sentiment, color in colors.items():
        plt.scatter([], [], c=color, label=sentiment)
    
    plt.title('Treasury Sentiment Over Time ')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_sentiment_score(df, threshold=0.25):

    # Sort dataframe by date index
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    
    # Plot hawkish score as line
    plt.plot(df_sorted.index, df_sorted['sentiment_score'], 'b.-', linewidth=1, alpha=0.7)

    # Add horizontal line
    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)

    # Add horizontal line
    plt.axhline(y=-threshold, color='black', linestyle='--', alpha=0.5)

    plt.title('Treasury Sentiment Score Over Time ')
    plt.xlabel('Date')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Treasury-specific fiscal policy terms (enhanced version)
EXPANSIONARY_TERMS = [
    'stimulus', 'spending', 'investment', 'expansion', 'growth', 'support',
    'boost', 'increase', 'recovery', 'relief', 'assistance', 'funding',
    'inject', 'accelerate', 'strengthen', 'enhance', 'expand', 'infrastructure',
    'jobs', 'employment', 'economic support', 'fiscal expansion', 'aid',
    'package', 'program', 'initiative', 'development', 'modernization',
    'create', 'build', 'establish', 'launch', 'deploy', 'allocate',
    'resources', 'capital', 'finance', 'grant', 'subsidy', 'incentive',
    'economic', 'productive', 'opportunity', 'prosperity', 'progress',
    'invest', 'fund', 'develop', 'promote', 'facilitate', 'stimulate',
    'revitalize', 'energize', 'mobilize', 'unleash',
    'public works', 'social programs', 'safety net',
    'procurement', 'contracting', 'hiring',
    'rebate', 'refund', 'tax credit',
    'liquidity', 'monetary accommodation',
    'competitive', 'innovation', 'research',
    'resilience', 'preparedness'
]

CONTRACTIONARY_TERMS = [
    'deficit', 'debt', 'reduction', 'cut', 'restraint', 'discipline',
    'consolidation', 'sustainable', 'prudent', 'efficiency', 'savings',
    'streamline', 'reduce', 'fiscal responsibility', 'balance', 'oversight',
    'accountability', 'cost control', 'budget', 'trim', 'eliminate',
    'reform', 'restructure', 'optimize', 'constraint', 'limitation',
    'ceiling', 'manage', 'control', 'limit', 'restrict', 'contain',
    'responsible', 'sound', 'stable', 'conservative', 'careful',
    'austerity', 'tighten', 'decrease', 'lower', 'minimize',
    'sequester', 'freeze', 'moratorium',
    'rationalize', 'downsize', 'rightsize',
    'fiscal cliff', 'unsustainable', 'burden',
    'taxpayer', 'waste', 'redundancy',
    'prioritize', 'focus', 'target',
    'review', 'audit', 'scrutiny',
    'cap', 'threshold'
]


def get_stance_label(score, threshold=0.25):
    """Convert score to stance with adjustable threshold
    
    Recommended thresholds for standardized scores:
    - 0.1: Very sensitive (captures subtle signals)
    - 0.5: Balanced approach (moderate confidence)
    - 1.0: Conservative (high confidence only)
    """
    if score > threshold:
        return 'expansionary'
    elif score < -threshold:
        return 'contractionary'
    else:
        return 'neutral'


def create_policy_axis(vectorizer):
    """Create fiscal policy axis with improved term matching"""

    feature_names = vectorizer.get_feature_names_out()
    
    # Find indices with more flexible matching
    expansionary_indices = []
    contractionary_indices = []
    
    for i, term in enumerate(feature_names):
        term_clean = term.lower().strip()
        
        # Check for matches (including partial matches for compound terms)
        exp_match = False
        con_match = False
        
        for exp_term in EXPANSIONARY_TERMS:
            if exp_term in term_clean or term_clean in exp_term:
                exp_match = True
                break
                
        for con_term in CONTRACTIONARY_TERMS:
            if con_term in term_clean or term_clean in con_term:
                con_match = True
                break
        
        # Avoid double-counting ambiguous terms
        if exp_match and not con_match:
            expansionary_indices.append(i)
        elif con_match and not exp_match:
            contractionary_indices.append(i)
        
    # Create fiscal axis
    axis = np.zeros(len(feature_names))
    
    # Weight terms by their importance
    for idx in expansionary_indices:
        axis[idx] = 1.0
        
    for idx in contractionary_indices:
        axis[idx] = -1.0
    
    # Normalize
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
        print(f"Fiscal axis normalized (norm was {norm:.3f})")
    else:
        print("⚠️  No fiscal terms found in vocabulary")
        # Create a fallback axis based on any available terms
        print("Creating fallback axis...")
        fallback_positive = ['growth', 'support', 'increase', 'development', 'program']
        fallback_negative = ['cut', 'reduce', 'control', 'limit', 'manage']
        
        for i, term in enumerate(feature_names):
            if any(pos in term.lower() for pos in fallback_positive):
                axis[i] = 0.5
            elif any(neg in term.lower() for neg in fallback_negative):
                axis[i] = -0.5
        
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
            print(f"Fallback axis created (norm: {norm:.3f})")
    
    return axis


def analyze_treasury_statements(statements_df, threshold=0.25):
    """Improved Treasury fiscal sentiment analysis"""

    if statements_df.empty:
        return pd.DataFrame(), None, None
    
    print(f"Analyzing {len(statements_df)} Treasury statements...")
    
    # Reset index to ensure clean indexing
    statements_df = statements_df.reset_index(drop=True)

    # Extract texts with better validation
    texts = []
    valid_indices = []
    
    for i, row in statements_df.iterrows():

        text = row.get('statement_text', '')

        if isinstance(text, str) and len(text.strip()) > 50:  # Require more substantial content

            texts.append(text)
            valid_indices.append(i)
    
    if not texts:
        print("❌ No valid text content found")
        return pd.DataFrame(), None, None
    
    try:
        # Create embeddings
        document_embeddings, vectorizer, processed_texts = create_document_embeddings(texts)
        
        # Create fiscal axis
        axis = create_policy_axis(vectorizer)
        
        # Calculate scores
        scores = calculate_sentiment_scores(document_embeddings, axis)
        
        # Create results - FIXED: Ensure we don't go out of bounds
        results = []

        for i in range(min(len(scores), len(valid_indices), len(processed_texts))):

            valid_idx = valid_indices[i]
            
            # Additional safety check
            if valid_idx >= len(statements_df):
                print(f"⚠️  Skipping index {valid_idx} (out of bounds)")
                continue
                
            row = statements_df.iloc[valid_idx]
            score = scores[i]
            stance = get_stance_label(score, threshold=threshold)
            date = row.get('date')

            results.append({'date': date, 'sentiment_score': score, 'stance': stance})
        
        results_df = pd.DataFrame(results)
                
        return results_df
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# Main analysis function that can be called from outside
def analyze_all_statements_embedding(statements_df, threshold=0.25, plot=False):
    """Main entry point for improved analysis"""

    # Apply embedding-based fiscal sentiment analysis
    print("Applying fiscal sentiment analysis...")

    # Run the improved analysis
    results_df = analyze_treasury_statements(statements_df, threshold=threshold)

    results_df.set_index('date', inplace=True)

    if plot:
        plot_sentiment(results_df)
        plot_sentiment_score(results_df, threshold=threshold)

    return results_df


def extract_clean_text(soup):
    """Extract clean, readable text from the page"""
    
    # Get the category (Press Releases)
    category = ""
    category_elem = soup.select_one('.news-category')
    if category_elem:
        category = category_elem.get_text().strip()
    
    # Get the title
    title = ""
    title_elem = soup.select_one('h2.uswds-page-title span')
    if title_elem:
        title = title_elem.get_text().strip()
    
    # Get the date
    date = ""
    date_elem = soup.select_one('.field--name-field-news-publication-date time')
    if date_elem:
        date_text = date_elem.get_text().strip()
        # Convert from "July 31, 2025" format if needed
        date = date_text
    
    # Get the main body text
    body_text = ""
    body_elem = soup.select_one('.field--name-field-news-body')
    if body_elem:
        # Extract text while preserving paragraph structure
        paragraphs = []
        for p in body_elem.find_all('p'):
            text = p.get_text().strip()
            if text:
                paragraphs.append(text)
        body_text = '\n\n'.join(paragraphs)
    
    # Combine all parts
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

    # Extract the main text content
    text_content = extract_clean_text(soup)

    return text_content


def scrape_text(prefix, num):
    """Scrape Treasury press release and return clean text content."""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    base_url = "https://home.treasury.gov/news/press-releases/"
    alt_base_url = "https://home.treasury.gov/news/press-release/"

    # Extract the main text content
    url = f"{base_url}{prefix}{num:04d}"
    text_content = extract_text(url, headers)

    if not text_content:
        # Extract the main text content
        url = f"{alt_base_url}{prefix}{num:04d}"
        text_content = extract_text(url, headers)

    if not text_content and len(str(abs(num))) == 3:

        # Extract the main text content
        url = f"{base_url}{prefix}{num:03d}"
        text_content = extract_text(url, headers)

        if not text_content:
            # Extract the main text content
            url = f"{alt_base_url}{prefix}{num:03d}"
            text_content = extract_text(url, headers)

    if not text_content:
        return None, None

    # First try the original pattern
    release_date = re.search(r'(\b\w+\s+\d{1,2},\s+\d{4})(?=\s*\(Archived Content\))', text_content)

    if not release_date:
        # Fallback: search in first 1000 characters
        early_text = text_content[:1000]
        release_date = re.search(r'(\b\w+\s+\d{1,2},\s+\d{4})', early_text)

    # Extract the date string from the match object
    release_date = release_date.group(1) if release_date else None

    if release_date is None:
        return None, None

    return text_content, release_date


def treasury_press_release_scraper(start_date, end_date, step=1):
    
    # Define Secretary eras with CORRECT starting statement numbers
    secretary_eras = [
        ( "jl", 2253, 2450), # Jack Lew era
        ( "jl", 2550, 2700), # Jack Lew era
        ( "jl", 9700, 9800), # Jack Lew era
        ( "jl", 9950, 10100), # Jack Lew era
        ( "jl", 40, 620), # Jack Lew era
        ("sm", 1, 1240),   # Mnuchin era - starts from sm0001  
        ("jy", 1, 2800),   # Yellen era - starts from jy0001
    ]

    statements_data = []

    for prefix, num_start, num_end in secretary_eras:

        for num in range(num_start, num_end + 1, step):

            text_content, release_date = scrape_text(prefix, num)

            if text_content is not None and release_date is not None:

                print(f'Scraped Treasury press release for {release_date}... ')

                statements_data.append({'date': release_date, 'statement_text': text_content})

    all_statements = pd.DataFrame(statements_data)
    all_statements['date'] = pd.to_datetime(all_statements['date'])
    all_statements = all_statements.sort_values('date')
    df_combined = all_statements.groupby('date')['statement_text'].apply(lambda x: '\n'.join(x)).reset_index()
    filtered_df = df_combined[(df_combined['date'] >= start_date) & (df_combined['date'] <= end_date)].copy()

    return filtered_df


def run_treasury_fiscal_analysis(start_date, end_date, step=1, plot=False):
    """Main function for Treasury fiscal sentiment analysis using embeddings"""

    statements_df = treasury_press_release_scraper(start_date, end_date, step=step)

    sentiment_df = analyze_all_statements_embedding(statements_df, plot=plot)

    return sentiment_df


# Example usage
if __name__ == "__main__":

    start_date = '2014-01-01'
    end_date = '2024-12-31'

    sentiment_df = run_treasury_fiscal_analysis(start_date, end_date, step=20, plot=True)
 