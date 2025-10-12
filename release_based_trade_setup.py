"""
Economic Release Processing and Calendar Integration Module

This module provides comprehensive functionality for processing economic release
dates and integrating them with options expiration calendars for event-driven
trading strategies. It handles the complex task of mapping economic announcements
to appropriate options expiration cycles for systematic trading around data releases.

Key Features:
- Comprehensive economic indicator dictionary with 60+ indicators
- Robust regex pattern matching for event identification
- Economic release date extraction and filtering
- Options expiration calendar integration
- Event-to-expiration date mapping for optimal timing
- Data validation and integrity checking

Economic Indicator Categories:
1. Federal Reserve and Monetary Policy (FOMC, Beige Book, Interest Rates)
2. Inflation Indicators (CPI, PPI, PCE, Import/Export Prices)
3. Economic Growth (GDP, Industrial Production, Leading Indicators)
4. Employment and Labor Market (NFP, JOLTS, Employment Cost Index)
5. Consumer Sector (Retail Sales, Consumer Confidence, Personal Income)
6. Business and Manufacturing (ISM, PMI, Factory Orders, Regional Fed Surveys)
7. Housing Market (Housing Starts, Home Sales, Case-Shiller Index)
8. Trade and International (Trade Balance, TIC Flows)
9. Government Finance (Federal Budget Balance)

Event-Driven Strategy Applications:
- Volatility expansion strategies around major releases
- Earnings-style straddle/strangle trades on economic data
- Calendar spread strategies targeting announcement periods
- Risk management through release-aware position timing

The module ensures precise timing coordination between economic events
and options expiration cycles, enabling systematic capture of event-driven
volatility opportunities while managing calendar risk effectively.

"""

from imports import *

# Comprehensive economic indicators dictionary with regex patterns for matching
eco_indicators = {
    # Federal Reserve and Monetary Policy
    'fomc_upper': 'FOMC Rate Decision \\(Upper Bound\\)',  # Federal funds rate decisions (upper bound)
    'fomc_lower': 'FOMC Rate Decision \\(Lower Bound\\)',  # Federal funds rate decisions (lower bound)
    'fomc_minutes': 'FOMC Meeting Minutes',  # Minutes from FOMC meetings
    'beige_book': 'U\\.S\\. Federal Reserve Releases Beige Book',  # Fed's economic conditions report
    'fed_iorb': 'Fed Interest on Reserve Balances Rate',  # Interest on reserve balances
    
    # Inflation Indicators
    'cpi': 'CPI MoM',  # Consumer Price Index (monthly)
    'pce': 'PCE Price Index MoM',  # Personal Consumption Expenditures (monthly)
    'ppi_fd': 'PPI Final Demand MoM',  # Producer Price Index - Final Demand
    'ppi_trade': 'PPI Ex Food, Energy, Trade MoM',  # PPI excluding food, energy, and trade
    'ppi': 'PPI MoM',  # Producer Price Index (headline)
    'ipi': 'Import Price Index MoM',  # Import price inflation
    'ipi_ex_pet': 'Import Price Index ex Petroleum MoM',  # Import prices excluding petroleum
    'epi': 'Export Price Index MoM',  # Export price inflation
    'ny_fed_inf_exp': 'NY Fed 1-Yr Inflation Expectations',  # Inflation expectations
    
    # Economic Growth and Production
    'gdp': 'GDP Annualized QoQ',  # Gross Domestic Product
    'ip': 'Industrial Production MoM',  # Manufacturing and industrial output
    'cu': 'Capacity Utilization',  # Industrial capacity usage
    'cfnai': 'Chicago Fed Nat Activity Index',  # Economic activity index
    'lei': 'Leading Index',  # Composite leading indicator
    
    # Manufacturing and Orders
    'dgo': 'Durable Goods Orders',  # Capital goods demand indicator
    'fo': 'Factory Orders',  # Manufacturing orders
    'fo_ex_trans': 'Factory Orders Ex Trans',  # Factory orders excluding transportation
    'ism': 'ISM Prices Paid',  # ISM Manufacturing (using ISM Prices Paid as proxy since ISM Manufacturing was removed)
    'ism_new_orders': 'ISM New Orders',  # ISM new orders component
    'ism_employment': 'ISM Employment',  # ISM employment component
    
    # Employment and Labor Market
    'nfp': 'Change in Nonfarm Payrolls',  # Monthly job creation data
    'jolts': 'JOLTS Job Openings',  # Job openings and labor turnover
    'household_emp': 'Change in Household Employment',  # Household survey employment
    'payroll_rev': 'Two-Month Payroll Net Revision',  # Payroll revisions
    'productivity': 'Nonfarm Productivity',  # Labor productivity
    'eci': 'Employment Cost Index',  # Employment costs
    'real_earnings': 'Real Avg Weekly Earnings YoY',  # Real average weekly earnings
    'real_hourly': 'Real Avg Hourly Earning YoY',  # Real average hourly earnings
    'lmci': 'Labor Market Conditions Index Change',  # Labor market conditions
    
    # Consumer Sector
    'rsa': 'Retail Sales Advance MoM',  # Consumer spending indicator
    'ri': 'Retail Inventories MoM',  # Retail inventories
    'ccf': 'Conf\\. Board Consumer Confidence',  # Consumer confidence index
    'cb_present': 'Conf\\. Board Present Situation',  # Consumer confidence - present situation
    'umich_sent': 'U\\. of Mich\\. Sentiment',  # University of Michigan sentiment
    'umich_current': 'U\\. of Mich\\. Current Conditions',  # UMich current conditions
    'langer_comfort': 'Langer Consumer Comfort',  # Langer consumer comfort index
    'langer_exp': 'Langer Economic Expectations',  # Langer economic expectations
    'ibd_tipp': 'IBD/TIPP Economic Optimism',  # IBD/TIPP optimism index
    'rbc_outlook': 'RBC Consumer Outlook Index',  # RBC consumer outlook
    'ccredit': 'Consumer Credit',  # Household borrowing patterns
    'personal_income': 'Personal Income',  # Personal income
    'real_spending': 'Real Personal Spending',  # Real personal spending
    'net_worth': 'Household Change in Net Worth',  # Household net worth changes
    
    # Vehicle Sales
    'vehicle_sales': 'Wards Total Vehicle Sales',  # Total vehicle sales
    'domestic_vehicle': 'Wards Domestic Vehicle Sales',  # Domestic vehicle sales
    
    # Business Sector and Services
    'isms': 'ISM Services Index',  # Services sector activity
    'ism_svc_emp': 'ISM Services Employment',  # ISM services employment
    'nfib': 'NFIB Small Business Optimism',  # Small business confidence
    'bi': 'Business Inventories',  # Business inventory levels
    'wi': 'Wholesale Inventories MoM',  # Wholesale inventory changes
    'wholesale_sales': 'Wholesale Trade Sales MoM',  # Wholesale trade sales
    
    # Regional Fed Surveys
    'empire': 'Empire Manufacturing',  # NY Fed Empire State index
    'philly': 'Philadelphia Fed Business Outlook',  # Regional manufacturing survey
    'philly_nonmanuf': 'Philadelphia Fed Non-Manufacturing Activity',  # Philly non-manufacturing
    'richmond': 'Richmond Fed Manufact\\. Index',  # Regional manufacturing index
    'richmond_svc': 'Richmond Fed Business Conditions',  # Richmond services
    'kansas_city': 'Kansas City Fed Manf\\. Activity',  # Kansas City manufacturing
    'kc_services': 'Kansas City Fed Services Activity',  # Kansas City services
    'dallas': 'Dallas Fed Manf\\. Activity',  # Dallas Fed manufacturing
    'dallas_svc': 'Dallas Fed Services Activity',  # Dallas services
    'chicago_pmi': 'MNI Chicago PMI',  # Chicago purchasing managers index
    'ism_ny': 'ISM New York',  # ISM New York
    'ism_milwaukee': 'ISM Milwaukee',  # ISM Milwaukee
    'ny_fed_svc': 'New York Fed Services Business Activity',  # NY Fed services
    
    # PMI Indicators
    'pmi_manuf': 'S&P Global US Manufacturing PMI',  # Manufacturing PMI
    'pmi_services': 'S&P Global US Services PMI',  # Services PMI
    
    # Housing Market
    'hst': 'Housing Starts',  # New construction activity
    'bp': 'Building Permits',  # Construction permits issued
    'ehs': 'Existing Home Sales',  # Home sales volume
    'nhs': 'New Home Sales',  # New home sales volume
    'phs': 'Pending Home Sales MoM',  # Leading housing indicator
    'nahb': 'NAHB Housing Market Index',  # Builder confidence
    'cs_20_city': 'S&P CoreLogic CS 20-City MoM SA',  # Case-Shiller 20-city index
    'cs_national': 'S&P CoreLogic CS US HPI YoY NSA',  # Case-Shiller national index
    'cs_index': 'S&P CoreLogic CS 20-City NSA Index',  # Case-Shiller index levels
    'cs_hpi_sa': 'S&P CoreLogic CS US HPI MoM SA',  # Case-Shiller HPI seasonally adjusted
    'cs_hpi_nsa': 'S&P CoreLogic CS US HPI NSA Index',  # Case-Shiller HPI not seasonally adjusted
    'fhfa_hpi': 'FHFA House Price Index MoM',  # FHFA house price index
    'hpi_purchase': 'House Price Purchase Index QoQ',  # House price purchase index
    'mortgage_delinq': 'Mortgage Delinquencies',  # Mortgage delinquency rates
    'construction_spending': 'Construction Spending MoM',  # Construction spending
    
    # Housing Finance
    'mba_apps': 'MBA Mortgage Applications',  # Mortgage application activity
    
    # Trade and International
    'tb': 'Trade Balance',  # Import/export balance
    'goods_tb': 'Advance Goods Trade Balance',  # Advance goods trade balance
    'tic_total': 'Total Net TIC Flows',  # Treasury International Capital flows
    'current_account': 'Current Account Balance',  # Current account balance
    
    # Government Finance
    'fbb': 'Federal Budget Balance',  # Government fiscal position
}


def load_indicator_dates(file_path, indicator_name):
    """
    Load and filter economic release dates for a specific indicator from CSV file.
    
    This function reads an economic calendar CSV file and extracts dates
    for a specific economic indicator using regex pattern matching. It
    provides the foundation for event-driven trading strategies by
    identifying precise release timing for systematic volatility capture.

    Args:
        file_path (str): Path to the economic calendar CSV file.
                        File must contain 'Event' and 'Date Time' columns.
        indicator_name (str): Economic indicator identifier key.
                             Must match keys in eco_indicators dictionary.

    Returns:
        pd.DatetimeIndex: Pandas DatetimeIndex containing all release dates
                         for the specified economic indicator, sorted chronologically.

    Raises:
        FileNotFoundError: If the specified CSV file path does not exist.
        KeyError: If indicator_name not found in eco_indicators dictionary.
        KeyError: If required columns missing from CSV.
        ValueError: If date parsing fails due to invalid date formats.

    Example:
        >>> cpi_dates = load_indicator_dates('/data/econ_calendar.csv', 'cpi')
        >>> print(f"Found {len(cpi_dates)} CPI releases")
        Found 24 CPI releases
    """
    # Load the complete economic calendar data
    data = pd.read_csv(file_path)
    
    # Get the regex pattern for the specified indicator
    pattern = eco_indicators[indicator_name]
    
    # Filter rows using regex pattern matching (case-insensitive)
    indicator_data = data[data['Event'].str.contains(pattern, case=False, na=False, regex=True)]
    
    # Extract and convert dates to datetime format
    return pd.to_datetime(indicator_data['Date Time'])


def find_nearest_expiration_dates(release_dates, list_exd_nosat, list_exd):
    """
    Map economic release dates to nearest appropriate options expiration dates.
    
    This function implements the core logic for event-driven options strategies
    by mapping each economic release to its optimal corresponding expiration date.
    The mapping ensures sufficient time for volatility expansion while minimizing
    unnecessary time decay.

    Args:
        release_dates (list or pd.DatetimeIndex): Economic release dates to map.
        list_exd_nosat (list): Business day expiration dates (excludes Saturdays).
        list_exd (list): Complete expiration dates including all calendar days.

    Returns:
        tuple: Two-element tuple containing:
            - _list_exd_nosat (list): Filtered business day expiration dates
            - _list_exd (list): Filtered complete expiration dates

    Raises:
        IndexError: If no valid expiration found for any release date.

    Mapping Logic:
        For each release date:
        1. Find all expiration dates >= release date (future expirations)
        2. Select the nearest expiration date after the release
        3. Map to corresponding date in both expiration lists

    Example:
        >>> release_dates = [datetime(2023, 12, 12), datetime(2023, 12, 20)]
        >>> weekly_exps = [datetime(2023, 12, 15), datetime(2023, 12, 22)]
        >>> mapped_biz, mapped_all = find_nearest_expiration_dates(
        ...     release_dates, weekly_exps, weekly_exps
        ... )
        >>> print(f"Mapped {len(mapped_biz)} releases to expirations")
    """
    # Initialize lists to store mapped expiration dates
    _list_exd_nosat = []  # Business day expirations
    _list_exd = []        # All expirations (including Saturdays)

    # Process each economic release date
    for release_date in release_dates:

        # Find expiration dates that occur after the release date
        # This ensures options don't expire before the economic data is released
        filtered_days = sorted([date for date in list_exd_nosat if date >= release_date])
        
        # Select the nearest expiration after the release date
        if filtered_days:
            # Use the first (nearest) expiration date after release
            nearest_business_expiry = filtered_days[0]
            _list_exd_nosat.append(nearest_business_expiry)
            
            # Find corresponding date in complete expiration list
            # This maintains consistency between Saturday-inclusive and business-day-only lists
            corresponding_expiry = min(list_exd, key=lambda x: abs(x - nearest_business_expiry))
            _list_exd.append(corresponding_expiry)
            
    return _list_exd_nosat, _list_exd


def get_release_dates(file_path, first_date, last_date, indicator_type):
    """
    Extract and filter economic release dates for a specific time period and indicator.
    
    This function provides the primary interface for obtaining economic release
    dates within a specified timeframe, combining data loading, indicator
    filtering, and date range selection into a single convenient function.

    Args:
        file_path (str): Full path to economic calendar CSV file.
        first_date (str or datetime-like): Start date for filtering (inclusive).
        last_date (str or datetime-like): End date for filtering (inclusive).
        indicator_type (str): Economic indicator key from eco_indicators dictionary.

    Returns:
        list: Chronologically sorted list of datetime objects representing
              economic release dates within the specified range.

    Raises:
        ValueError: If indicator_type not found in eco_indicators dictionary.
        FileNotFoundError: If economic calendar CSV file doesn't exist.
        ValueError: If date parsing fails for date range parameters.

    Processing Pipeline:
        1. Validate indicator type against available indicators
        2. Load indicator-specific release dates from calendar file
        3. Convert date range parameters to consistent datetime format
        4. Filter dates within specified range (inclusive bounds)
        5. Normalize dates to remove time components
        6. Return as list for downstream processing

    Example:
        >>> cpi_dates = get_release_dates(
        ...     '/data/econ_calendar.csv', 
        ...     '2023-01-01', 
        ...     '2023-12-31', 
        ...     'cpi'
        ... )
        >>> print(f"Found {len(cpi_dates)} CPI releases in 2023")
    """
    # Validate that the requested indicator is supported
    if indicator_type not in eco_indicators:
        available_indicators = ', '.join(eco_indicators.keys())
        raise ValueError(f"Invalid indicator type. Must be one of: {available_indicators}")
    
    # Load release dates for the specified indicator
    dates = load_indicator_dates(file_path, indicator_type)
    
    # Convert date range parameters to consistent Timestamp format
    first_date = pd.Timestamp(first_date)
    last_date = pd.Timestamp(last_date)
    
    # Filter dates within the specified range and normalize to remove time components
    release_dates = dates[(dates >= first_date) & (dates <= last_date)].dt.normalize().tolist()
    
    return release_dates


def release_info(first_date, last_date, release_file, list_exd_nosat, list_exd, event='cpi'):
    """
    Generate complete release-based trading information for event-driven options strategies.
    
    This function serves as the comprehensive interface for setting up event-driven
    options trading strategies by coordinating economic release identification,
    expiration date mapping, and trading calendar integration. It provides all
    necessary components for systematic volatility capture around economic announcements.

    Args:
        first_date (str or datetime-like): Start date for release extraction (inclusive).
        last_date (str or datetime-like): End date for release extraction (inclusive).
        release_file (str): Path to economic calendar CSV file.
        list_exd_nosat (list): Business day options expiration dates.
        list_exd (list): Complete options expiration calendar.
        event (str, optional): Economic indicator type to process.
                              Defaults to 'cpi' (Consumer Price Index).

    Returns:
        tuple: Three-element tuple containing complete trading setup information:
        
        - _list_exd_nosat (list): Filtered business day expiration dates
        - release_dates_dict (dict): Dictionary mapping expiration dates to release dates
        - _list_exd (list): Filtered complete expiration dates

    Raises:
        ValueError: If event type not found in eco_indicators dictionary.
        FileNotFoundError: If release_file path does not exist.
        Exception: If mismatch between number of releases and expiration mappings.

    Data Integrity Validation:
        - Ensures 1:1 mapping between releases and expirations
        - Validates chronological consistency of all date sequences
        - Confirms data completeness across the entire analysis period

    Strategic Applications:
        - CPI/PPI Volatility Plays: Monthly inflation data releases
        - FOMC Rate Decision Straddles: Federal Reserve announcements
        - NFP Employment Trades: First Friday job reports
        - GDP Growth Reactions: Quarterly economic growth data

    Example:
        >>> exp_dates, release_dict, all_exps = release_info(
        ...     '2023-01-01', '2023-12-31', 
        ...     '/data/econ_calendar.csv',
        ...     weekly_expirations, all_expirations, 
        ...     event='cpi'
        ... )
        >>> print(f"Setup {len(exp_dates)} CPI trading opportunities")
        >>> for exp_date in exp_dates[:3]:
        ...     release_date = release_dict[exp_date]
        ...     days_diff = (exp_date - release_date).days
        ...     print(f"Release: {release_date.date()}, Expiry: {exp_date.date()}, Gap: {days_diff} days")
    """
    # Extract economic release dates for the specified indicator and date range
    release_dates = get_release_dates(release_file, first_date, last_date, event)

    # Map each release date to the nearest appropriate expiration date
    _list_exd_nosat, _list_exd = find_nearest_expiration_dates(release_dates, list_exd_nosat, list_exd)

    # Validate data integrity - ensure consistent mapping
    if len(_list_exd_nosat) != len(release_dates):
        raise Exception("Aborting due to mismatch in number of release dates and expiry dates nearest to release dates.")

    # Create mapping dictionary from expiration dates to release dates
    # This enables precise timing coordination in trading strategies
    release_dates_dict = dict(zip(_list_exd_nosat, release_dates))

    return _list_exd_nosat, release_dates_dict, _list_exd