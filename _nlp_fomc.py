from imports import *
from release_based_trade_setup import get_release_dates
from _finbert import load_bert_model, analyze_statement_sentiment


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
    
    plt.title('FOMC Sentiment Over Time (FinBERT-FOMC)')
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
        
    plt.title('FOMC Sentiment Score Over Time (FinBERT-FOMC)')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_all_statements(statements_df, plot=False):
    """Apply BERT to all statements"""

    sentiment_model = load_bert_model()
    results = []
    
    for idx, row in statements_df.iterrows():

        print(f"Analyzing statement {row['date']}...")
        
        raw_label, sentiment_score = analyze_statement_sentiment(row['statement_text'], sentiment_model)
        
        if raw_label is not None:

            results.append({'date': row['date'], 'sentiment_label': raw_label, 'sentiment_score': sentiment_score})

    results_df = pd.DataFrame(results)

    results_df['date'] = pd.to_datetime(results_df['date'])

    results_df.set_index('date', inplace=True)

    if plot:
        plot_sentiment(results_df)
        plot_sentiment_score(results_df)

    return results_df


def scrape_statement(date_str):
    """Scrape FOMC statement for a given date"""

    # Correct Fed URL pattern
    base_url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary"
    
    # Convert date format: "2024-12-18" -> "20241218"
    formatted_date = date_str.replace('-', '')
    
    # Fed URL pattern: monetary + date + 'a.htm'
    url = f"{base_url}{formatted_date}a.htm"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content (Fed website structure)
            content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
            if content_div:
                # Get all paragraphs - this is the statement text
                paragraphs = content_div.find_all('p')
                statement_text = ' '.join([p.get_text().strip() for p in paragraphs])
                return statement_text
            
        print(f"Failed to fetch statement for {date_str}")
        return None
        
    except Exception as e:
        print(f"Error scraping {date_str}: {e}")
        return None


def scrape_statements_from_release_dates(release_dates):
    """Scrape FOMC statements using your existing release dates"""

    statements_data = []
    
    for date_str in release_dates:
        print(f"Scraping FOMC statement for {date_str}...")
        
        statement_text = scrape_statement(date_str)
        
        if statement_text:
            statements_data.append({'date': date_str, 'statement_text': statement_text})
        
        time.sleep(1)  # Be polite to Fed website
    
    return pd.DataFrame(statements_data)


def run_fomc_monetary_analysis(first_date, last_date, plot=False):
    """Main function using your existing release dates"""

    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, '..', 'data')
    release_file = os.path.join(data_directory, 'economic_calendar_010112_123124.csv')
    event = 'fomc_upper'
    release_dates = get_release_dates(release_file, first_date, last_date, event)
    release_dates = [ts.strftime("%Y-%m-%d") for ts in release_dates]

    # Scrape FOMC statements using your exact dates
    print("Scraping FOMC statements...")
    statements_df = scrape_statements_from_release_dates(release_dates)
    
    # Apply BERT analysis
    print("Applying BERT sentiment analysis...")
    sentiment_df = analyze_all_statements(statements_df, plot=plot)

    return sentiment_df


# Example usage
if __name__ == "__main__":

    start_date = '2014-01-01'
    end_date = '2024-12-31'

    sentiment_df = run_fomc_monetary_analysis(start_date, end_date, plot=True)
 