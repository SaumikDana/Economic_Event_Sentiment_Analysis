from imports import *
from _embeddings import create_document_embeddings, calculate_sentiment_scores


def plot_sentiment(df):

    # Sort dataframe by date index
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    # Create scatter plot
    plt.figure(figsize=(14, 9))
    colors = {'negative': 'red', 'positive': 'green', 'neutral': 'gray'}
    
    # Plot each point in chronological order
    for i, (date, sentiment) in enumerate(zip(df_sorted.index, df_sorted['sentiment_label'])):
        plt.scatter(date, sentiment, c=colors[sentiment], alpha=0.7, s=20)
    
    # Add legend manually
    for sentiment, color in colors.items():
        plt.scatter([], [], c=color, label=sentiment)

    plt.legend()  # <-- This was missing    
    company = df['company'].iloc[0]
    plt.title(f'{company} Sentiment Over Time ')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_sentiment_score(df):
    """Plot the sentiment score over time as a line chart"""

    # Sort dataframe by date index
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    
    # Plot hawkish score as line
    plt.plot(df_sorted.index, df_sorted['sentiment_score'], 'b.-', linewidth=2, alpha=0.7)

    company = df['company'].iloc[0]
    plt.title(f'{company} Sentiment Score Over Time ')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Corporate event sentiment terms (enhanced version)
POSITIVE_CORPORATE_TERMS = [
    'growth', 'expansion', 'increase', 'strong', 'positive', 'improved',
    'exceeded', 'beat', 'outperformed', 'successful', 'achievement', 'milestone',
    'record', 'robust', 'solid', 'momentum', 'accelerate', 'enhance',
    'opportunity', 'progress', 'innovation', 'leadership',
    'partnership', 'acquisition', 'merger', 'synergies', 'efficiency',
    'margin', 'profit', 'revenue', 'earnings', 'dividend', 'buyback',
    'investment', 'development', 'launch', 'approval', 'patent',
    'breakthrough', 'market share', 'demand',
    'resilient', 'capitalize', 'outpace', 'surge', 'rally', 'upgrade', 'premium', 'diversify',
    'scale', 'penetrate', 'unlock', 'deliver', 'execute', 'capture', 'expand', 'strengthen',
    'accelerating', 'flourishing', 'thriving', 'prosperous', 'promising',
    'bullish', 'upbeat', 'optimistic', 'confident', 'favorable',
    'rebound', 'recovery', 'upturn', 'turnaround', 'revival',
    'franchise', 'moat', 'differentiated', 'best-in-class', 'world-class'
]

NEGATIVE_CORPORATE_TERMS = [
    'decline', 'decrease', 'weak', 'negative', 'disappointing', 'missed',
    'below', 'underperformed', 'challenging', 'difficult', 'concern',
    'risk', 'uncertainty', 'volatility', 'pressure', 'headwind',
    'investigation', 'litigation', 'settlement', 'fine', 'penalty',
    'restructuring', 'layoff', 'closure', 'discontinue', 'impairment',
    'writedown', 'loss', 'deficit', 'shortfall', 'delay', 'postpone',
    'suspend', 'recall', 'warning', 'guidance', 'lower', 'reduce',
    'cut', 'departure', 'resignation', 'termination',
    'breach', 'violation', 'default', 'bankruptcy', 'insolvency',
    'fraud', 'scandal', 'misconduct', 'whistleblower', 'subpoena',
    'downturn', 'recession', 'slump', 'correction', 'crash',
    'deteriorate', 'erode', 'plunge', 'plummet', 'tumble',
    'struggle', 'suffer', 'falter', 'stumble', 'setback',
    'crisis', 'emergency', 'catastrophe', 'disaster', 'collapse',
    'bearish', 'pessimistic', 'cautious', 'concerned', 'worried',
    'disruption', 'threat', 'competition', 'obsolete', 'outdated',
    'regulatory', 'compliance', 'audit', 'probe', 'inquiry',
    'vulnerable', 'exposed', 'at-risk', 'troubled', 'distressed'
]

# Configuration
RATE_LIMIT = 0.11  # Stay under 10 requests/second
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Your Company) your-email@domain.com',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'www.sec.gov'
}
BASE_URL = "https://www.sec.gov"


def get_cik_from_ticker(ticker):
    """Get CIK (Central Index Key) from ticker symbol"""
    
    try:
        # SEC company tickers JSON endpoint
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(tickers_url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        data = response.json()
        
        # Search for ticker (case insensitive)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                cik = str(entry['cik_str']).zfill(10)  # Pad with zeros to 10 digits
                print(f"Found CIK {cik} for ticker {ticker} ({entry['title']})")
                return cik
        
        print(f"⚠️  Ticker {ticker} not found in SEC database")
        return None
        
    except Exception as e:
        print(f"Error looking up ticker {ticker}: {e}")
        return None


def get_sentiment_label(score, threshold=0.25):
    """Convert score to sentiment label"""
    
    if score > threshold:
        return 'positive'
    elif score < -threshold:
        return 'negative'
    else:
        return 'neutral'


def create_corporate_axis(vectorizer):
    """Create corporate sentiment axis (adapted from your create_policy_axis)"""
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Find indices with flexible matching
    positive_indices = []
    negative_indices = []
    
    for i, term in enumerate(feature_names):
        term_clean = term.lower().strip()
        
        # Check for matches
        pos_match = False
        neg_match = False
        
        for pos_term in POSITIVE_CORPORATE_TERMS:
            if pos_term in term_clean or term_clean in pos_term:
                pos_match = True
                break
                
        for neg_term in NEGATIVE_CORPORATE_TERMS:
            if neg_term in term_clean or term_clean in neg_term:
                neg_match = True
                break
        
        # Avoid double-counting ambiguous terms
        if pos_match and not neg_match:
            positive_indices.append(i)
        elif neg_match and not pos_match:
            negative_indices.append(i)
    
    # Create corporate sentiment axis
    axis = np.zeros(len(feature_names))
    
    # Weight terms by their importance
    for idx in positive_indices:
        axis[idx] = 1.0
        
    for idx in negative_indices:
        axis[idx] = -1.0
    
    # Normalize
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
    else:
        print("⚠️  No corporate terms found in vocabulary")
        # Create fallback axis
        fallback_positive = ['growth', 'increase', 'strong', 'positive', 'success']
        fallback_negative = ['decline', 'decrease', 'weak', 'negative', 'loss']
        
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


def extract_8k_text(filing_url):
    """Extract text content from 8-K filing"""
    
    try:
        # Get the filing page
        response = requests.get(filing_url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        # Find the actual filing document link
        content = response.text
        
        # Look for .txt or .htm file link
        txt_match = re.search(r'href="([^"]*\.txt)"', content)
        htm_match = re.search(r'href="([^"]*\.htm)"', content)
        
        doc_link = None
        if txt_match:
            doc_link = urljoin(BASE_URL, txt_match.group(1))
        elif htm_match:
            doc_link = urljoin(BASE_URL, htm_match.group(1))
        
        if not doc_link:
            print(f"Could not find document link for {filing_url}")
            return ""
        
        # Get the actual document
        doc_response = requests.get(doc_link, headers=HEADERS)
        doc_response.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        # Clean up the text
        text = doc_response.text
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from {filing_url}: {e}")
        return ""


def test_sec_api_response(cik):
    """Test what the SEC API actually returns"""
    
    print(f"Testing SEC API response format for CIK {cik}...")
    
    # Try the new JSON API first
    try:
        cik_padded = str(cik).zfill(10)
        json_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        print(f"Trying JSON API: {json_url}")
        
        response = requests.get(json_url, headers=HEADERS)
        print(f"JSON API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("JSON API Success!")
            print(f"Company: {data.get('name', 'Unknown')}")
            
            filings = data.get('filings', {}).get('recent', {})
            forms = filings.get('form', [])
            dates = filings.get('filingDate', [])
            
            # Count 8-K filings
            eight_k_count = sum(1 for form in forms if form == '8-K')
            print(f"Total 8-K filings found: {eight_k_count}")
            
            # Show recent 8-K filings
            recent_8ks = [(form, date) for form, date in zip(forms, dates) if form == '8-K'][:5]
            print("Recent 8-K filings:")
            for form, date in recent_8ks:
                print(f"  {date}: {form}")
            
            return True
            
    except Exception as e:
        print(f"JSON API failed: {e}")
    
    # Try browse-edgar
    try:
        search_url = f"{BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': '8-K',
            'dateb': '20241231',
            'count': '40'
        }
        
        print(f"Trying browse-edgar: {search_url}")
        response = requests.get(search_url, params=params, headers=HEADERS)
        print(f"Browse-edgar Status: {response.status_code}")
        print(f"Response length: {len(response.content)}")
        
        if response.status_code == 200:
            content = response.text
            print("First 500 chars of response:")
            print(content[:500])
            print("\n" + "="*50 + "\n")
            
            # Look for common patterns
            if "8-K" in content:
                print("✓ Found '8-K' in response")
            else:
                print("✗ No '8-K' found in response")
                
            if "filing-date" in content:
                print("✓ Found 'filing-date' in response")
            else:
                print("✗ No 'filing-date' found in response")
                
            if "<entry>" in content:
                print("✓ Found '<entry>' XML tags")
            else:
                print("✗ No '<entry>' XML tags found")
                
        return response.status_code == 200
        
    except Exception as e:
        print(f"Browse-edgar failed: {e}")
        return False


def search_8k_filings_alternative(cik, start_date, end_date):
    """Alternative method using SEC submissions API"""
    
    print(f"Using SEC JSON API for CIK {cik}...")
    
    # Format CIK with leading zeros (10 digits total)
    cik_padded = str(cik).zfill(10)
    
    # Get company submissions data - correct URL format
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    try:
        # Add proper headers for SEC API
        headers = {
            'User-Agent': 'Mozilla/5.0 (Your Company) your-email@domain.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        data = response.json()
        print(f"✅ Successfully retrieved data for {data.get('name', 'Unknown Company')}")
        
        filings = []
        recent_filings = data.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            print("No recent filings data found")
            return []
        
        forms = recent_filings.get('form', [])
        dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Convert date strings to date objects for comparison
        start_dt = datetime.datetime.strptime(start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime.date) else start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime.date) else end_date, "%Y-%m-%d").date()
        
        print(f"Searching for 8-K filings between {start_dt} and {end_dt}")
        print(f"Total forms in recent filings: {len(forms)}")
        
        # Count all 8-K forms first
        eight_k_total = sum(1 for form in forms if form == '8-K')
        print(f"Total 8-K forms found: {eight_k_total}")
        
        for i, (form, date_str, accession) in enumerate(zip(forms, dates, accession_numbers)):
            if form == '8-K':
                try:
                    filing_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    if start_dt <= filing_date <= end_dt:
                        # Construct filing URL - remove dashes from accession number for path
                        accession_clean = accession.replace('-', '')
                        # Use the CIK without leading zeros for the path
                        cik_clean = str(int(cik))
                        filing_url = f"{BASE_URL}/Archives/edgar/data/{cik_clean}/{accession_clean}/{accession}-index.html"
                        
                        filing = {
                            'company_name': data.get('name', 'Unknown'),
                            'filing_url': filing_url,
                            'date': date_str,
                            'filing_date': filing_date,
                            'accession': accession
                        }
                        filings.append(filing)
                        
                except Exception as e:
                    print(f"Error processing filing {i}: {e}")
                    continue
        
        # Sort by date (most recent first)
        filings.sort(key=lambda x: x['filing_date'], reverse=True)
        
        print(f"JSON API found {len(filings)} 8-K filings in date range")
        
        # Debug: show first few filings
        if filings:
            print("Sample filings found:")
            for filing in filings[:5]:
                print(f"  📄 {filing['date']}: {filing['filing_url']}")
        
        return filings
        
    except Exception as e:
        print(f"JSON API search failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def search_8k_filings_by_cik_and_date(cik, start_date, end_date, max_results=100):
    """Search for 8-K filings by CIK and date range using SEC EDGAR search"""
    
    print(f"Searching 8-K filings for CIK {cik} from {start_date} to {end_date}...")
    
    # Try alternative method first (more reliable)
    filings = search_8k_filings_alternative(cik, start_date, end_date)
    
    if filings:
        return filings[:max_results]  # Limit results
    
    # Fallback to original method if alternative fails
    print("Trying original search method...")
    
    # Convert dates to YYYYMMDD format
    start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime.date) else start_date.replace("-", "")
    end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime.date) else end_date.replace("-", "")
    
    # Build search URL - use the browse interface
    search_url = f"{BASE_URL}/cgi-bin/browse-edgar"
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': '8-K',
        'dateb': end_str,  # Before this date
        'count': max_results,
        'output': 'xml'
    }
    
    try:
        response = requests.get(search_url, params=params, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT)
        
        print(f"Response status: {response.status_code}")
        print(f"Response length: {len(response.content)}")
        
        # Try to parse as XML first, if that fails try HTML parsing
        filings = []
        try:
            root = ET.fromstring(response.content)
            
            # Look for entries in the XML
            for entry in root.findall('.//entry'):
                try:
                    filing_date = entry.find('filing-date').text
                    filing_href = entry.find('filing-href').text
                    company_name = entry.find('company-name').text
                    
                    # Convert filing date to datetime for comparison
                    filing_dt = datetime.datetime.strptime(filing_date, "%Y-%m-%d").date()
                    start_dt = datetime.datetime.strptime(start_str, "%Y%m%d").date() if isinstance(start_date, str) else start_date
                    end_dt = datetime.datetime.strptime(end_str, "%Y%m%d").date() if isinstance(end_date, str) else end_date
                    
                    # Filter by date range
                    if start_dt <= filing_dt <= end_dt:
                        filing = {
                            'company_name': company_name,
                            'filing_url': filing_href,
                            'date': filing_date,
                            'filing_date': filing_dt
                        }
                        filings.append(filing)
                        
                except Exception as e:
                    print(f"Error parsing filing entry: {e}")
                    continue
                    
        except ET.ParseError:
            # If XML parsing fails, try HTML parsing approach
            print("XML parsing failed, trying HTML approach...")
            
            # Try different approach - get HTML and parse manually
            params_html = {
                'action': 'getcompany',
                'CIK': cik,
                'type': '8-K',
                'dateb': end_str,
                'count': max_results
            }
            
            response_html = requests.get(search_url, params=params_html, headers=HEADERS)
            response_html.raise_for_status()
            time.sleep(RATE_LIMIT)
            
            # Parse HTML content for filings table
            content = response_html.text
            
            # Look for filings table entries
            # Pattern: <tr><td>[date]</td><td><a href="[link]">8-K</a></td>...
            filing_pattern = r'<tr[^>]*>.*?<td[^>]*>(\d{4}-\d{2}-\d{2})</td>.*?<td[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>8-K</a>'
            matches = re.findall(filing_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    filing_date_str, filing_href = match
                    filing_dt = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                    start_dt = datetime.datetime.strptime(start_str, "%Y%m%d").date() if isinstance(start_date, str) else start_date
                    end_dt = datetime.datetime.strptime(end_str, "%Y%m%d").date() if isinstance(end_date, str) else end_date
                    
                    if start_dt <= filing_dt <= end_dt:
                        # Get company name from the CIK lookup we did earlier
                        filing = {
                            'company_name': 'Apple Inc.',  # We know this from CIK lookup
                            'filing_url': urljoin(BASE_URL, filing_href),
                            'date': filing_date_str,
                            'filing_date': filing_dt
                        }
                        filings.append(filing)
                        
                except Exception as e:
                    print(f"Error parsing HTML filing entry: {e}")
                    continue
        
        # Sort by date (most recent first)
        filings.sort(key=lambda x: x['filing_date'], reverse=True)
        
        print(f"Found {len(filings)} 8-K filings in date range")
        
        # Debug: print first few filings if found
        if filings:
            print("Sample filings found:")
            for filing in filings[:3]:
                print(f"  {filing['date']}: {filing['filing_url']}")
        
        return filings
        
    except Exception as e:
        print(f"Error searching filings: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_8k_filings_by_ticker_and_date(ticker, start_date, end_date, threshold=0.25, plot=False):
    """
    Analyze 8-K filings for a specific ticker and date range
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        start_date (str or datetime): Start date in 'YYYY-MM-DD' format
        end_date (str or datetime): End date in 'YYYY-MM-DD' format
        threshold (float): Sentiment threshold for classification
    
    Returns:
        pd.DataFrame: Analysis results with sentiment scores
    """
    
    print("=== SEC 8-K Analysis by Ticker and Date Range ===")
    print(f"Ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Sentiment Threshold: {threshold}\n")
    
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Get CIK for ticker
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return pd.DataFrame()
    
    # Search for filings
    filings = search_8k_filings_by_cik_and_date(cik, start_date, end_date)
    
    if not filings:
        print(f"No 8-K filings found for {ticker} in the specified date range")
        return pd.DataFrame()
    
    # Extract text from filings
    texts = []
    valid_filings = []
    
    for i, filing in enumerate(filings):
        print(f"Processing {i+1}/{len(filings)}: {filing['company_name']} - {filing['date']}")
        
        text = extract_8k_text(filing['filing_url'])
        
        if text and len(text.strip()) > 100:  # Require substantial content
            texts.append(text)
            valid_filings.append(filing)
        else:
            print(f"Skipping {filing['date']} - insufficient text content")
    
    if not texts:
        print("No valid text content found")
        return pd.DataFrame()
    
    try:
        # Create embeddings
        document_embeddings, vectorizer, processed_texts = create_document_embeddings(texts)
        
        # Create corporate sentiment axis
        axis = create_corporate_axis(vectorizer)
        
        # Calculate sentiment scores
        scores = calculate_sentiment_scores(document_embeddings, axis)
        
        # Create results
        results = []
        for i, (filing, score) in enumerate(zip(valid_filings, scores)):
            label = get_sentiment_label(score, threshold)
            result = {
                'date': filing['date'],
                'ticker': ticker.upper(),
                'company': filing['company_name'],
                'sentiment_score': round(score, 4),
                'sentiment_label': label
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.sort_values('date', ascending=False)
        
        # Print summary
        print(f"\n=== Analysis Summary ===")
        print(f"Total 8-K filings analyzed: {len(results_df)}")
        sentiment_counts = results_df['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment.capitalize()} filings: {count}")
        
        if len(results_df) > 0:
            avg_sentiment = results_df['sentiment_score'].mean()
            print(f"Average sentiment score: {avg_sentiment:.4f}")
        
        results_df.set_index('date', inplace=True)

        if plot:
            plot_sentiment(results_df)
            plot_sentiment_score(results_df)

        return results_df
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return pd.DataFrame()


# Main execution
if __name__ == "__main__":
    
    underlying = 'AAPL'
    start_date = '2014-01-01'
    end_date = '2024-12-31'

    results = analyze_8k_filings_by_ticker_and_date(underlying, start_date, end_date, plot=True)
    
    
