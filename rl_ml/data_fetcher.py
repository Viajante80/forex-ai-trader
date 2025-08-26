"""
Oanda Data Fetcher for RL Trading Agent
Fetches historical data with ask, bid, mid prices and volume
"""
import requests
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import logging
from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OandaDataFetcher:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update(self.config.headers)
        
    def fetch_candles(self, 
                     instrument: str = None,
                     granularity: str = None,
                     from_time: str = None,
                     to_time: str = None,
                     count: int = 5000) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data from Oanda
        """
        instrument = instrument or self.config.instrument
        granularity = granularity or self.config.timeframe
        
        url = f"{self.config.base_url}/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "price": "BAM",  # Bid, Ask, Mid
        }
        
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        if count is not None:
            params["count"] = count
            
        try:
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return None
                
            data = response.json()
            
            if "candles" not in data:
                logger.error("No candles data in response")
                logger.error(f"Response: {data}")
                return None
                
            return self._process_candles(data["candles"])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _process_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Process raw candle data into pandas DataFrame
        """
        processed_data = []
        
        for candle in candles:
            if not candle.get("complete", False):
                continue
                
            timestamp = pd.to_datetime(candle["time"])
            volume = int(candle["volume"])
            
            # Extract bid prices
            bid = candle["bid"]
            bid_open = float(bid["o"])
            bid_high = float(bid["h"])
            bid_low = float(bid["l"])
            bid_close = float(bid["c"])
            
            # Extract ask prices
            ask = candle["ask"]
            ask_open = float(ask["o"])
            ask_high = float(ask["h"])
            ask_low = float(ask["l"])
            ask_close = float(ask["c"])
            
            # Extract mid prices
            mid = candle["mid"]
            mid_open = float(mid["o"])
            mid_high = float(mid["h"])
            mid_low = float(mid["l"])
            mid_close = float(mid["c"])
            
            # Calculate spread
            spread_open = ask_open - bid_open
            spread_high = ask_high - bid_high
            spread_low = ask_low - bid_low
            spread_close = ask_close - bid_close
            
            processed_data.append({
                "timestamp": timestamp,
                "bid_open": bid_open,
                "bid_high": bid_high,
                "bid_low": bid_low,
                "bid_close": bid_close,
                "ask_open": ask_open,
                "ask_high": ask_high,
                "ask_low": ask_low,
                "ask_close": ask_close,
                "mid_open": mid_open,
                "mid_high": mid_high,
                "mid_low": mid_low,
                "mid_close": mid_close,
                "volume": volume,
                "spread_open": spread_open,
                "spread_high": spread_high,
                "spread_low": spread_low,
                "spread_close": spread_close,
            })
        
        df = pd.DataFrame(processed_data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def fetch_historical_data(self, 
                            start_date: str = None,
                            end_date: str = None,
                            instrument: str = None,
                            granularity: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch complete historical data between two dates
        Uses count-based fetching to work around Oanda API limitations
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.test_end_date
        instrument = instrument or self.config.instrument
        granularity = granularity or self.config.timeframe
        
        logger.info(f"Fetching {instrument} data from {start_date} to {end_date}")
        
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        all_data = []
        
        # Start from the end date and work backwards using count-based requests
        current_end_dt = end_dt
        
        while current_end_dt > start_dt:
            to_time = current_end_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            
            logger.info(f"Fetching chunk ending at: {to_time}")
            
            # Fetch maximum allowed candles (5000) working backwards
            chunk_data = self.fetch_candles(
                instrument=instrument,
                granularity=granularity,
                to_time=to_time,
                count=5000
            )
            
            if chunk_data is not None and not chunk_data.empty:
                # Filter data to only include what's after our start date
                chunk_data = chunk_data[chunk_data.index >= start_dt]
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    logger.info(f"Fetched {len(chunk_data)} candles (filtered)")
                    
                    # Move to the earliest timestamp we got
                    current_end_dt = chunk_data.index[0] - timedelta(minutes=5)
                else:
                    logger.info("All data filtered out (before start date)")
                    break
            else:
                logger.warning(f"No data received for chunk ending at {to_time}")
                # Move back by estimated time period
                current_end_dt -= timedelta(days=30)
            
            # Rate limiting
            time.sleep(0.2)
            
            # Safety check to prevent infinite loops
            if current_end_dt < start_dt - timedelta(days=365):
                logger.warning("Reached reasonable limit, stopping fetch")
                break
        
        if not all_data:
            logger.error("No data fetched")
            return None
            
        # Combine all data and sort
        combined_data = pd.concat(all_data).drop_duplicates().sort_index()
        
        # Final filter to ensure we're within the requested date range
        combined_data = combined_data[
            (combined_data.index >= start_dt) & 
            (combined_data.index < end_dt)
        ]
        
        logger.info(f"Total candles fetched: {len(combined_data)}")
        logger.info(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
        
        return combined_data
    
    def save_data(self, df: pd.DataFrame, filename: str = None, data_type: str = "raw"):
        """
        Save data to pickle file for efficient storage
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./data/{self.config.instrument}_{self.config.timeframe}_{data_type}_{timestamp}.pkl"
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_pickle(filename)
        logger.info(f"📋 {data_type.capitalize()} data saved to {filename}")
        logger.info(f"   - Columns: {len(df.columns)}")
        logger.info(f"   - Rows: {len(df):,} candles") 
        logger.info(f"   - Date range: {df.index[0]} to {df.index[-1]}")
        
        return filename
    
    def load_data(self, filename: str = None) -> Optional[pd.DataFrame]:
        """
        Load data from pickle file
        """
        if filename is None:
            filename = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_data.pkl"
        
        try:
            df = pd.read_pickle(filename)
            logger.info(f"Data loaded from {filename}: {len(df)} candles")
            return df
        except FileNotFoundError:
            logger.warning(f"File {filename} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

def main():
    """Main entry point for data fetching CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch Forex data from Oanda')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--timeframe', default='M5', help='Timeframe')
    parser.add_argument('--start-date', default='2016-01-01', help='Start date')
    parser.add_argument('--end-date', default='2025-01-01', help='End date')
    parser.add_argument('--sample', action='store_true', help='Fetch sample data only')
    
    args = parser.parse_args()
    
    # Update config
    from config import CONFIG
    CONFIG.instrument = args.instrument
    CONFIG.timeframe = args.timeframe
    CONFIG.start_date = args.start_date
    CONFIG.test_end_date = args.end_date
    
    fetcher = OandaDataFetcher()
    
    if args.sample:
        print("Fetching sample data...")
        data = fetcher.fetch_candles(count=100)
        if data is not None:
            print("Sample data:")
            print(data.head())
            print(f"Shape: {data.shape}")
    else:
        print(f"Fetching historical data for {args.instrument} from {args.start_date} to {args.end_date}")
        data = fetcher.fetch_historical_data()
        if data is not None:
            # Save only raw data
            raw_filename = fetcher.save_data(data, data_type="raw")
            print(f"✅ Raw data saved: {raw_filename}")
            print(f"   📊 Columns: {len(data.columns)}")
            print(f"   📈 Candles: {len(data):,}")
            print(f"\n💡 To add technical indicators, run:")
            print(f"   uv run python technical_indicators.py {raw_filename}")

if __name__ == "__main__":
    main()