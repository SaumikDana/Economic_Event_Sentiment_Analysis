from imports import *


def plot_sentiment(df):
    """Plot employment sentiment over time"""
    
    # Sort dataframe by date index
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    # Create scatter plot
    plt.figure(figsize=(14, 9))
    colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}
    
    # Plot each point in chronological order
    for i, (date, sentiment) in enumerate(zip(df_sorted.index, df_sorted['stance'])):
        plt.scatter(date, sentiment, c=colors[sentiment], alpha=0.7, s=20)
    
    # Add legend manually
    for sentiment, color in colors.items():
        plt.scatter([], [], c=color, label=sentiment)
    
    plt.title('BLS Employment Sentiment Over Time (Enhanced Data-Driven)')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.legend()
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
    
    # Plot sentiment score as line
    plt.plot(df_sorted.index, df_sorted['sentiment_score'], 'b.-', linewidth=2, alpha=0.7)
        
    plt.title('BLS Employment Sentiment Score Over Time (Enhanced Data-Driven)')
    plt.xlabel('Date')
    plt.ylabel('Employment Sentiment Score')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Create sentiment labels based on threshold
def get_sentiment_label(score, threshold):

    if score > threshold:
        return 'positive'
    elif score < -threshold:
        return 'negative' 
    else:
        return 'neutral'


def get_sentiment_components_weights(df):

    # Create enhanced composite sentiment score with dynamic weighting
    sentiment_components = []
    weights = []
    
    # Core indicators (always present)
    # Unemployment component (negative change is good)
    sentiment_components.append(-df['unemployment_change'])
    weights.append(0.25)  # 25% weight
    
    # Optional indicators (add if available)
    if 'payroll_change' in df.columns and df['payroll_change'].notna().any():
        payroll_scaled = df['payroll_change'] / 1000  # Scale down large numbers
        sentiment_components.append(payroll_scaled)
        weights.append(0.20)  # 20% weight
    
    if 'wage_change' in df.columns and df['wage_change'].notna().any():
        sentiment_components.append(df['wage_change'])
        weights.append(0.15)  # 15% weight
    
    if 'participation_change' in df.columns and df['participation_change'].notna().any():
        sentiment_components.append(df['participation_change'])
        weights.append(0.15)  # 15% weight
    
    if 'underemployment_change' in df.columns and df['underemployment_change'].notna().any():
        # Negative change in underemployment is good
        sentiment_components.append(-df['underemployment_change'])
        weights.append(0.15)  # 15% weight
    
    if 'job_openings_change' in df.columns and df['job_openings_change'].notna().any():
        sentiment_components.append(df['job_openings_change'])
        weights.append(0.10)  # 10% weight

    return sentiment_components, weights


def get_new_columns(df):

    # Calculate month-over-month changes for all available indicators
    df['unemployment_change'] = df['unemployment_rate'].diff()
    
    if 'nonfarm_payrolls' in df.columns:
        df['payroll_change'] = df['nonfarm_payrolls'].diff()
    
    if 'avg_hourly_earnings' in df.columns:
        df['wage_change'] = df['avg_hourly_earnings'].pct_change()
    
    if 'labor_force_participation' in df.columns:
        df['participation_change'] = df['labor_force_participation'].diff()
    
    if 'underemployment_rate' in df.columns:
        df['underemployment_change'] = df['underemployment_rate'].diff()
    
    if 'job_openings' in df.columns:
        df['job_openings_change'] = df['job_openings'].pct_change()

    return df


def calculate_employment_sentiment_from_data(df, threshold=0.15):
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df = get_new_columns(df)
    
    sentiment_components, weights = get_sentiment_components_weights(df)
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
        
    # Combine components
    if sentiment_components:
        df['sentiment_score'] = sum(comp * weight for comp, weight in zip(sentiment_components, normalized_weights))
    else:
        df['sentiment_score'] = 0
    
    # Fill missing values
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
        
    # Apply with lambda to pass both arguments
    df['stance'] = df['sentiment_score'].apply(lambda x: get_sentiment_label(x, threshold))

    # Select relevant columns for output (include all available indicators)
    output_cols = ['date', 'sentiment_score', 'stance', 'unemployment_rate']
    
    # Add all available change indicators
    for col in df.columns:
        if col.endswith('_change') or col in ['nonfarm_payrolls', 'labor_force_participation', 'underemployment_rate', 'job_openings']:
            if col not in output_cols:
                output_cols.append(col)
    
    # Filter to only include columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    
    result_df = df[available_cols].copy()
    result_df = result_df.dropna(subset=['sentiment_score']).set_index('date')
    
    return result_df


def get_bls_data_via_api(series_ids, start_year, end_year):
    
    headers = {'Content-type': 'application/json'}
    
    # Use FREE API endpoint (no key required)
    url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
    data = {
        'seriesid': series_ids,
        'startyear': str(start_year),
        'endyear': str(end_year)
    }
    
    try:
        # Make API request
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=30)
        
        if response.status_code == 200:

            json_data = response.json()
            
            if json_data['status'] == 'REQUEST_SUCCEEDED':

                results = []
                
                for series in json_data['Results']['series']:

                    series_id = series['seriesID']
                    
                    for item in series['data']:
                        # Convert period to proper date
                        if item['period'].startswith('M'):
                            month = item['period'].replace('M', '').zfill(2)
                            date_str = f"{item['year']}-{month}-01"
                        else:
                            date_str = f"{item['year']}-01-01"
                        
                        result = {
                            'series_id': series_id,
                            'year': int(item['year']),
                            'period': item['period'],
                            'value': float(item['value']) if item['value'] != '' else None,
                            'date': date_str
                        }
                        results.append(result)
                
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()


def get_employment_data_bls_api(start_year=2020, end_year=2024):
    """Get comprehensive employment data from BLS API (NO KEY REQUIRED)"""
        
    # Enhanced employment series IDs - now pulling 6 key indicators
    key_series = [
        'LNS14000000',   # Unemployment Rate 
        'CES0000000001', # Total Nonfarm Payrolls
        'CES0500000003', # Average Hourly Earnings
        'LNS11300000',   # Labor Force Participation Rate
        'LNS13327709',   # U-6 Underemployment Rate
        'JTS000000000000000JOL'  # JOLTS Job Openings (if available)
    ]
    
    series_names = {
        'LNS14000000': 'unemployment_rate',
        'CES0000000001': 'nonfarm_payrolls', 
        'CES0500000003': 'avg_hourly_earnings',
        'LNS11300000': 'labor_force_participation',
        'LNS13327709': 'underemployment_rate',
        'JTS000000000000000JOL': 'job_openings'
    }
    
    df = get_bls_data_via_api(key_series, start_year, end_year)
    
    if not df.empty:
        # Pivot to have series as columns
        df_pivot = df.pivot_table(
            index='date', 
            columns='series_id', 
            values='value', 
            aggfunc='first'
        ).reset_index()
        
        # Rename columns to friendly names
        df_pivot = df_pivot.rename(columns=series_names)
        
        # Sort by date
        df_pivot = df_pivot.sort_values('date')
        
        return df_pivot

    else:
        print("❌ Failed to retrieve data from BLS API")
        return pd.DataFrame()


def run_bls_employment_analysis(start_year=2020, end_year=2024, threshold=0.15, plot=False):
    
    # Get employment data from BLS API
    employment_data = get_employment_data_bls_api(start_year, end_year)
            
    # Calculate sentiment from data trends
    sentiment_df = calculate_employment_sentiment_from_data(employment_data, threshold)
            
    if plot:
        plot_sentiment(sentiment_df)
        plot_sentiment_score(sentiment_df)
            
    return sentiment_df


# Example usage - ENHANCED VERSION WITH MORE INDICATORS
if __name__ == "__main__":
        
    # Run the enhanced analysis
    results = run_bls_employment_analysis(
        start_year=2014, 
        end_year=2024, 
        threshold=0.12,  # Slightly lower threshold for more sensitivity
        plot=True
    )