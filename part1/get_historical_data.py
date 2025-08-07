import pandas as pd
import os
import time
from datetime import timezone, datetime
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountInstruments
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

# Check if credentials are loaded properly
if not API_KEY or not ACCOUNT_ID:
    raise ValueError("Missing OANDA credentials. Ensure OANDA_API_KEY and OANDA_ACCOUNT_ID are in your .env file.")

# Define major currency pairs
# major_pairs = [
#     "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", 
#     "AUD_USD", "USD_CAD", "NZD_USD"
# ]

major_pairs = [
    "EUR_USD", "GBP_USD", "EUR_GBP"]

# Define timeframes to download
timeframes = ["M5","M15", "M30", "H1", "H4", "D", "W"]

# Start date for historical data
start_date = datetime(2016, 1, 1)

# Create a data directory if it doesn't exist
data_dir = "../oanda_historical_data"
os.makedirs(data_dir, exist_ok=True)

# Initialize the API client with credentials from .env
client = oandapyV20.API(access_token=API_KEY)

def get_instrument_details():
    """
    Retrieve instrument details including pipLocation from OANDA API
    Returns a dictionary with instrument name as key and instrument details as value
    """
    try:
        # Create and execute request to get instrument details
        request = AccountInstruments(accountID=ACCOUNT_ID)
        client.request(request)
        
        # Process the response into a dictionary for easier lookup
        instruments_dict = {}
        for instrument in request.response['instruments']:
            name = instrument['name']
            instruments_dict[name] = instrument
            
        return instruments_dict
        
    except Exception as e:
        print(f"Error retrieving instrument details: {e}")
        return {}

def get_candles_df(instrument, response, pip_location):
    """
    Convert API response to pandas DataFrame with normalized prices
    """
    prices = []
    # Calculate pip size based on pip location
    # pipLocation is 10^x where x is the pipLocation value
    # For EUR/USD with pipLocation = -4, pip_size = 0.0001 (1 pip)
    pip_size = 10 ** pip_location
    
    for candle in response['candles']:
        if candle['complete']:
            # Original price values
            open_price = float(candle['mid']['o'])
            high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l'])
            close_price = float(candle['mid']['c'])
            
            # Normalized price values (convert to pips)
            # price_in_pips = price / pip_size
            norm_open = open_price / pip_size
            norm_high = high_price / pip_size
            norm_low = low_price / pip_size
            norm_close = close_price / pip_size
            
            prices.append({
                'time': candle['time'],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': int(candle['volume']),
                'norm_open': norm_open,
                'norm_high': norm_high,
                'norm_low': norm_low,
                'norm_close': norm_close
            })
    return pd.DataFrame(prices)

def get_historical_data(instrument, start_date, timeframe, pip_location):
    """
    Download historical OANDA data from start_date to now
    - Handles OANDA's 5000 candle limit with pagination
    - Returns a pandas DataFrame with all data including normalized values
    """
    print(f"Downloading {instrument} {timeframe} data from {start_date}...")
    
    # Convert start_date to RFC3339 format
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    
    # Format start date for OANDA API
    from_time = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Initialize an empty DataFrame to store all candles
    all_data = pd.DataFrame()
    
    # Initial params for first request
    params = {
        "from": from_time,
        "granularity": timeframe,
        "count": 5000  # Maximum allowed by OANDA
    }
    
    # Keep track of the latest time received
    latest_time = from_time
    
    # Loop until we've collected all available data
    more_data = True
    request_count = 0
    
    while more_data:
        try:
            # Create and send the request
            r = InstrumentsCandles(instrument=instrument, params=params)
            client.request(r)
            
            # Increment request counter
            request_count += 1
            
            # Convert response candles to DataFrame with normalized values
            candles_df = get_candles_df(instrument, r.response, pip_location)
            
            # If we got data back
            if not candles_df.empty:
                # Add this batch to our main DataFrame
                all_data = pd.concat([all_data, candles_df])
                
                # Get the time of the last candle for the next iteration
                latest_time = candles_df['time'].iloc[-1]
                
                # Update params for next request with the new "from" time
                params["from"] = latest_time
                
                print(f"  Downloaded batch {request_count}: {len(candles_df)} candles ending at {latest_time}")
                
                # If we got fewer than 5000 candles, we've reached the end
                if len(candles_df) < 5000:
                    more_data = False
            else:
                # No data returned, we're done
                more_data = False
            
            # Respect API rate limits
            if more_data and request_count % 5 == 0:
                print("  Pausing for API rate limits...")
                time.sleep(1)  # Add a delay to respect OANDA's rate limits
                
        except Exception as e:
            print(f"  Error: {e}")
            print("  Pausing and trying again...")
            time.sleep(5)  # Longer pause on error
            
            # If we've had too many errors, give up
            if request_count > 3:
                more_data = False
    
    # Clean up the DataFrame
    if not all_data.empty:
        # Remove duplicates that might occur at pagination boundaries
        all_data = all_data.drop_duplicates(subset=['time'])
        
        # Convert time strings to datetime objects
        all_data['time'] = pd.to_datetime(all_data['time'])
        
        # Set the time column as index
        all_data.set_index('time', inplace=True)
        
        # Sort by time
        all_data.sort_index(inplace=True)
    
    return all_data

def calculate_additional_features(df, instrument_details):
    """
    Calculate additional normalized features for ML/RL
    - Returns DataFrame with additional features
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Extract relevant details from instrument properties
    display_precision = instrument_details.get('displayPrecision', 5)
    
    # 1. Calculate normalized returns (percentage changes)
    df_features['return'] = df_features['close'].pct_change() * 100
    df_features['norm_return'] = df_features['norm_close'].pct_change() * 100
    
    # 2. Calculate pip differences for high-low range
    df_features['pip_range'] = df_features['norm_high'] - df_features['norm_low']
    
    # 3. Calculate pip movement from open to close
    df_features['pip_move'] = df_features['norm_close'] - df_features['norm_open']
    
    # 4. Volatility estimate (standard deviation of returns over rolling window)
    df_features['volatility_10'] = df_features['norm_return'].rolling(window=10).std()
    
    # 5. Normalized body size (relative to instrument's typical range)
    df_features['body_size'] = abs(df_features['norm_close'] - df_features['norm_open'])
    
    # 6. Upper and lower shadows (wicks) in pips
    df_features['upper_shadow'] = df_features['norm_high'] - df_features['norm_close'].where(
        df_features['norm_close'] >= df_features['norm_open'], 
        df_features['norm_open']
    )
    
    df_features['lower_shadow'] = df_features['norm_open'].where(
        df_features['norm_close'] >= df_features['norm_open'], 
        df_features['norm_close']
    ) - df_features['norm_low']
    
    # Drop NaN values created by calculations
    df_features.dropna(inplace=True)
    
    return df_features

# Get instrument details at the beginning
print("Retrieving instrument details from OANDA...")
instruments_dict = get_instrument_details()

if not instruments_dict:
    raise ValueError("Failed to retrieve instrument details. Cannot proceed with normalization.")

# Main execution loop
for pair in major_pairs:
    # Create a pair-specific directory
    pair_dir = os.path.join(data_dir, pair)
    os.makedirs(pair_dir, exist_ok=True)
    
    # Get pip location for this instrument
    # pipLocation is typically -4 for most major pairs (like EUR/USD),
    # meaning 1 pip = 0.0001, so we need to multiply by 10^4 to normalize
    instrument_details = instruments_dict.get(pair, {})
    pip_location = instrument_details.get('pipLocation', -4)
    
    print(f"\nProcessing {pair} with pipLocation: {pip_location}")
    print(f"1 pip = {10**pip_location}")
    
    for timeframe in timeframes:
        try:
            # Get historical data with normalized values
            historical_data = get_historical_data(pair, start_date, timeframe, pip_location)
            
            if not historical_data.empty:
                # Calculate additional features for ML/RL
                enhanced_data = calculate_additional_features(historical_data, instrument_details)
                
                # Save to pickle format
                file_path = os.path.join(pair_dir, f"{pair}_{timeframe}_normalized.pkl")
                enhanced_data.to_pickle(file_path)
                
                # Display summary
                print(f"\nSuccessfully downloaded and normalized {pair} {timeframe} data:")
                print(f"  Total candles: {len(enhanced_data)}")
                print(f"  Date range: {enhanced_data.index.min()} to {enhanced_data.index.max()}")
                print(f"  Saved to: {file_path}\n")
                
                # Calculate file size for user information
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  File size: {file_size_mb:.2f} MB")
                
                # Save a sample CSV for reference (first 100 rows)
                sample_path = os.path.join(pair_dir, f"{pair}_{timeframe}_sample.csv")
                enhanced_data.head(100).to_csv(sample_path)
                print(f"  Sample saved to: {sample_path}")
            else:
                print(f"\nNo data available for {pair} {timeframe}\n")
                
            # Add a delay between different timeframes to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"\nFailed to download {pair} {timeframe} data: {e}\n")
    
    # Add a longer delay between different pairs to respect rate limits
    print(f"Completed downloads for {pair}. Pausing before next pair...\n")
    time.sleep(3)

print("All downloads completed!")