import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
import traceback

from _embeddings import (
    create_document_embeddings, 
    calculate_sentiment_scores,
    create_policy_axis
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


def plot_sentiment_score(df, threshold=0.25):
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


def get_stance_label(score, threshold=0.25):
    """Convert score to stance"""
    if score > threshold:
        return 'expansionary'
    elif score < -threshold:
        return 'contractionary'
    else:
        return 'neutral'


def analyze_treasury_statements(statements_df, labeled_examples, threshold=0.25):
    """Treasury fiscal sentiment analysis using labeled examples
    
    Args:
        statements_df: DataFrame with columns ['date', 'statement_text']
        labeled_examples: Dict with keys 'expansionary' and 'contractionary', 
                         each containing list of labeled Treasury statements
        threshold: Threshold for stance classification
    """

    if statements_df.empty:
        return pd.DataFrame()
    
    if labeled_examples is None:
        raise ValueError("Must provide labeled_examples with 'expansionary' and 'contractionary' keys")
    
    if 'expansionary' not in labeled_examples or 'contractionary' not in labeled_examples:
        raise ValueError("labeled_examples must have 'expansionary' and 'contractionary' keys")
    
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
        # Create embeddings
        document_embeddings, model, processed_texts = create_document_embeddings(texts)
        
        # Create policy axis from YOUR labeled examples
        axis = create_policy_axis(
            model,
            labeled_examples['expansionary'],
            labeled_examples['contractionary']
        )
        
        # Calculate scores
        scores = calculate_sentiment_scores(document_embeddings, axis)
        
        # Create results
        results = []
        for i in range(min(len(scores), len(valid_indices))):
            valid_idx = valid_indices[i]
            
            if valid_idx >= len(statements_df):
                continue
                
            row = statements_df.iloc[valid_idx]
            score = scores[i]
            stance = get_stance_label(score, threshold=threshold)
            date = row.get('date')

            results.append({
                'date': date, 
                'sentiment_score': score, 
                'stance': stance
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"\n✓ Analysis complete:")
        print(f"  Expansionary: {(results_df['stance'] == 'expansionary').sum()}")
        print(f"  Neutral: {(results_df['stance'] == 'neutral').sum()}")
        print(f"  Contractionary: {(results_df['stance'] == 'contractionary').sum()}")
                
        return results_df
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def analyze_all_statements(statements_df, labeled_examples, threshold=0.25, plot=False):
    """Main entry point for sentiment analysis"""

    print("\n" + "="*80)
    print("TREASURY FISCAL SENTIMENT ANALYSIS")
    print("="*80)

    results_df = analyze_treasury_statements(statements_df, labeled_examples, threshold=threshold)

    if not results_df.empty:
        results_df.set_index('date', inplace=True)

        if plot:
            plot_sentiment(results_df)
            plot_sentiment_score(results_df, threshold=threshold)

    return results_df


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


def treasury_press_release_scraper(start_date, end_date, step=1):
    
    secretary_eras = [
        ("jl", 2253, 2450),
        ("jl", 2550, 2700),
        ("jl", 9700, 9800),
        ("jl", 9950, 10100),
        ("jl", 40, 620),
        ("sm", 1, 1240),
        ("jy", 1, 2800),
    ]

    statements_data = []

    for prefix, num_start, num_end in secretary_eras:
        for num in range(num_start, num_end + 1, step):
            text_content, release_date = scrape_text(prefix, num)
            if text_content is not None and release_date is not None:
                print(f'Scraped: {release_date}')
                statements_data.append({'date': release_date, 'statement_text': text_content})

    all_statements = pd.DataFrame(statements_data)
    all_statements['date'] = pd.to_datetime(all_statements['date'])
    all_statements = all_statements.sort_values('date')
    df_combined = all_statements.groupby('date')['statement_text'].apply(lambda x: '\n'.join(x)).reset_index()
    filtered_df = df_combined[(df_combined['date'] >= start_date) & (df_combined['date'] <= end_date)].copy()

    return filtered_df


def run_treasury_fiscal_analysis(start_date, end_date, labeled_examples, step=1, plot=False):

    statements_df = treasury_press_release_scraper(start_date, end_date, step=step)
    sentiment_df = analyze_all_statements(statements_df, labeled_examples, plot=plot)

    return sentiment_df


if __name__ == "__main__":
    
    labeled_examples = {
        'expansionary': [
            """The Treasury Department today announced the rapid deployment of emergency fiscal support to stabilize households, workers, and businesses. These measures are designed to inject liquidity into the economy, preserve employment, and prevent a deeper contraction during the ongoing crisis.""",

            """In coordination with Congress, Treasury is implementing large-scale fiscal relief to support aggregate demand and ensure continued access to credit markets. These actions are intended to sustain economic activity and accelerate recovery.""",

            """The Administration is committed to using the full capacity of federal fiscal policy to support economic growth, expand public investment, and strengthen the resilience of the U.S. economy through targeted spending initiatives.""",

            """Treasury will continue to support expansionary fiscal measures that provide direct assistance to families, state and local governments, and small businesses to mitigate economic headwinds and promote broad-based recovery.""",

            """This legislation authorizes significant increases in federal outlays to address infrastructure needs, public health investments, and workforce support, reinforcing near-term demand and long-term productive capacity.""",

            """The Department supports fiscal actions that increase government spending during periods of economic slack to counteract private sector retrenchment and stabilize employment.""",

            """Treasury’s fiscal strategy emphasizes accommodative budgetary policy to sustain growth, reduce unemployment, and prevent deflationary pressures from taking hold in the broader economy.""",

            """The proposed fiscal package reflects an expansionary stance aimed at boosting consumption, supporting investment, and ensuring that the recovery remains durable and inclusive."""
        ],
        'contractionary': [
            """The Treasury Department emphasizes the importance of fiscal discipline and long-term debt sustainability, including measures to restrain discretionary spending and reduce structural deficits.""",

            """This agreement reflects a commitment to fiscal consolidation through spending caps, entitlement reform, and deficit reduction to ensure confidence in U.S. public finances.""",

            """Treasury supports policies that reduce federal borrowing and place the budget on a sustainable path by limiting expenditure growth and enhancing revenue adequacy.""",

            """The Administration is focused on restoring fiscal balance by constraining outlays and addressing long-term obligations that pose risks to economic stability.""",

            """This framework prioritizes deficit reduction and debt stabilization, recognizing that excessive fiscal expansion could undermine market confidence and long-term growth.""",

            """Treasury endorses a measured fiscal approach that avoids further stimulus and instead emphasizes budgetary restraint to contain inflationary and debt-related risks.""",

            """The Department supports fiscal reforms that slow the growth of federal spending and reduce reliance on deficit financing over the medium term.""",

            """This legislation advances fiscal responsibility by enforcing spending limits and promoting a gradual return to primary budget balance."""
        ]
    }

    sentiment_df = run_treasury_fiscal_analysis(
        start_date='2014-01-01',
        end_date='2024-12-31',
        labeled_examples=labeled_examples,
        step=20,
        plot=True
    )
