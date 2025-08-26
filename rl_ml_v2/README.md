# RL Forex Trading System - UV Setup & Execution Guide

## 1. Prerequisites & UV Installation

### Install UV (if not already installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

### Verify UV Installation
```bash
uv --version
```

## 2. Project Setup

### Create Project Directory
```bash
mkdir forex-rl-trading
cd forex-rl-trading
```

### Initialize UV Project
```bash
# Initialize new UV project
uv init

# Or if you prefer a specific Python version
uv init --python 3.11
```

## 3. Dependencies Installation

### Create requirements.txt
```bash
# Create requirements file for the RL forex system
cat > requirements.txt << EOF
# Core ML/RL libraries
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
gymnasium>=0.29.0
scikit-learn>=1.3.0

# Technical Analysis
ta>=0.10.2
TA-Lib>=0.4.28

# API and Data
requests>=2.31.0
oandapyV20>=0.7.2

# Visualization and Analysis
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Utilities
python-dateutil>=2.8.0
pytz>=2023.3
tqdm>=4.65.0
dataclasses-json>=0.6.0
pyyaml>=6.0

# Optional: Jupyter for analysis
jupyter>=1.0.0
ipykernel>=6.25.0
EOF
```

### Install Dependencies with UV
```bash
# Install all dependencies
uv pip install -r requirements.txt

# Or install individually for better control
uv pip install torch numpy pandas gymnasium scikit-learn
uv pip install ta requests oandapyV20 matplotlib plotly
```

## 4. Code Setup

### Save the Main Code
Save the RL forex trading system code as `forex_rl_trader.py`:

```bash
# Create the main trading system file
touch forex_rl_trader.py
```

**Copy the complete RL forex trading system code into this file.**

### Create Configuration File
```bash
# Create config file
cat > config.yaml << EOF
# OANDA Configuration
oanda:
  api_key: "your_oanda_api_key_here"
  account_id: "your_account_id_here"
  environment: "practice"  # or "live"

# Trading Parameters
trading:
  instruments: ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
  max_position_size: 0.02
  risk_per_trade: 0.01
  max_daily_loss: 0.06
  max_drawdown: 0.10

# RL Parameters
rl:
  learning_rate: 0.0003
  batch_size: 256
  memory_size: 100000
  training_episodes: 1000

# Technical Indicators
indicators:
  rsi_periods: [9, 14, 25]
  ema_periods: [10, 21, 50, 100]
  obv_period: 14
  fibonacci_periods: [20, 50, 100]
  pivot_period: "D"
EOF
```

### Create Execution Scripts

#### Quick Test Script
```bash
cat > run_quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test script for RL forex system"""

import sys
import os
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_rl_trader import *

def quick_test():
    """Run a quick test with minimal training"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuration for quick testing
    config = TradingConfig(
        # Update with your OANDA credentials
        oanda_api_key="your_api_key",
        oanda_account_id="your_account_id", 
        oanda_environment="practice",
        
        # Reduced parameters for quick testing
        training_episodes=50,  # Reduced for quick test
        batch_size=128,
        memory_size=10000,
        
        # Standard trading parameters
        max_position_size=0.02,
        risk_per_trade=0.01,
    )
    
    logger.info("=== QUICK TEST MODE ===")
    
    # Initialize trader
    trader = ForexRLTrader(config)
    
    # Quick training test (recent data only)
    logger.info("Testing data preparation...")
    test_data = trader.prepare_data("EUR_USD", "2024-01-01", "2024-03-01")
    
    if not test_data.empty:
        logger.info(f"Data loaded successfully: {len(test_data)} samples")
        logger.info(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Test environment creation
        env = ForexTradingEnvironment(test_data, config)
        logger.info("Environment created successfully")
        
        # Test agent
        state = env.reset()
        action = trader.agent.act(state)
        logger.info(f"Agent test successful - Action shape: {action.shape}")
        
        # Quick training (5 episodes)
        logger.info("Running mini training session...")
        trader.train_on_historical_data("EUR_USD", "2024-01-01", "2024-02-01", episodes=5)
        
        logger.info("✅ Quick test completed successfully!")
        return True
    else:
        logger.error("❌ Data loading failed - check OANDA credentials")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 System is ready for full execution!")
        print("Next steps:")
        print("1. Update OANDA credentials in the script")
        print("2. Run: uv run run_full_analysis.py")
    else:
        print("\n❌ Setup issues detected. Please check configuration.")
EOF
```

#### Full Analysis Script
```bash
cat > run_full_analysis.py << 'EOF'
#!/usr/bin/env python3
"""Full RL forex analysis with 2016-2024 training and 2024-2025 testing"""

import sys
import os
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_rl_trader import *

def main():
    """Run full walk-forward analysis"""
    
    # Production configuration
    config = TradingConfig(
        # IMPORTANT: Update with your OANDA credentials
        oanda_api_key="YOUR_OANDA_API_KEY_HERE",
        oanda_account_id="YOUR_ACCOUNT_ID_HERE",
        oanda_environment="practice",
        
        # Optimized parameters from research
        instruments=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'],
        max_position_size=0.02,
        risk_per_trade=0.01,
        max_daily_loss=0.06,
        max_drawdown=0.10,
        
        # SAC parameters
        learning_rate=3e-4,
        batch_size=256,
        memory_size=100000,
        training_episodes=800,  # Full training
        
        # Enhanced technical indicators
        rsi_periods=[9, 14, 25],
        ema_periods=[10, 21, 50, 100],
        obv_period=14,
        fibonacci_periods=[20, 50, 100],
        pivot_period="D"
    )
    
    print("=" * 60)
    print("🚀 FOREX RL TRADING SYSTEM - FULL ANALYSIS")
    print("=" * 60)
    print(f"Training Period: 2016-2024")
    print(f"Testing Period: 2024-2025") 
    print(f"Algorithm: Soft Actor-Critic (SAC)")
    print(f"Timeframe: 15-minute")
    print(f"Indicators: RSI, EMA, MACD, BB, OBV, Fibonacci, Pivots")
    print("=" * 60)
    
    # Initialize trader
    trader = ForexRLTrader(config)
    
    # Run comprehensive analysis
    try:
        results = trader.run_full_walk_forward_analysis("EUR_USD")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"forex_rl_results_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_results[key] = value
            except TypeError:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {results_file}")
        
        # Print summary
        if 'overall_summary' in results:
            summary = results['overall_summary']
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"   Average Return: {summary.get('avg_return', 0):.2%}")
            print(f"   Average Sharpe: {summary.get('avg_sharpe', 0):.2f}")
            print(f"   Max Drawdown: {summary.get('avg_max_drawdown', 0):.2%}")
            print(f"   Win Rate: {summary.get('avg_win_rate', 0):.2%}")
            print(f"   Consistency: {summary.get('consistency', 0):.2%}")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
EOF
```

## 5. OANDA Account Setup

### Get OANDA Credentials
1. Go to [OANDA](https://www.oanda.com) and create a practice account
2. Login to your account
3. Navigate to "Manage API Access"
4. Generate API key and note your Account ID
5. Update the credentials in your scripts

### Test OANDA Connection
```bash
cat > test_oanda.py << 'EOF'
import requests

def test_oanda_connection(api_key, account_id):
    """Test OANDA API connection"""
    
    base_url = "https://api-fxpractice.oanda.com"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test account info
        response = requests.get(f"{base_url}/v3/accounts/{account_id}", headers=headers)
        response.raise_for_status()
        
        account_data = response.json()['account']
        print(f"✅ OANDA Connection Successful!")
        print(f"Account Balance: {account_data.get('balance', 'N/A')}")
        print(f"Currency: {account_data.get('currency', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ OANDA Connection Failed: {e}")
        return False

# Update with your credentials
API_KEY = "your_api_key_here"
ACCOUNT_ID = "your_account_id_here"

if __name__ == "__main__":
    test_oanda_connection(API_KEY, ACCOUNT_ID)
EOF
```

## 6. Running the System

### Step 1: Quick Test
```bash
# Update credentials in run_quick_test.py, then run:
uv run run_quick_test.py
```

### Step 2: Full Analysis (after quick test passes)
```bash
# Update credentials in run_full_analysis.py, then run:
uv run run_full_analysis.py
```

### Step 3: Monitor Progress
The system will output detailed logs showing:
- Data loading progress
- Training episode progress  
- Performance metrics
- Walk-forward validation results

### Step 4: View Results
Results will be saved as JSON files with comprehensive performance metrics.

## 7. Alternative Execution Methods

### Using UV with Scripts
```bash
# Run specific functions
uv run -c "from forex_rl_trader import *; print('System loaded')"

# Interactive mode
uv run python -i forex_rl_trader.py
```

### Using UV with Jupyter
```bash
# Install and run Jupyter
uv pip install jupyter
uv run jupyter notebook

# Then create a new notebook and import the system
```

## 8. Troubleshooting

### Common Issues

#### OANDA API Issues
```bash
# Test connection
uv run test_oanda.py
```

#### Package Installation Issues  
```bash
# Clean install
uv pip uninstall torch numpy pandas
uv pip install torch numpy pandas --no-cache-dir
```

#### Memory Issues (Large Training)
```bash
# Run with reduced parameters
# Edit training_episodes=200 in config
```

## 9. Production Deployment

### For Live Trading
1. Switch `oanda_environment` to "live" 
2. Use live OANDA credentials
3. Implement additional monitoring
4. Set up automated execution

### Performance Monitoring
```bash
# Monitor system resources during training
htop  # or top on macOS
```

## 10. Expected Runtime

- **Quick Test**: 2-5 minutes
- **Full Analysis**: 2-4 hours (depending on hardware)
- **Data Loading**: 10-30 minutes (depends on internet speed)
- **Training**: 1-3 hours per instrument
- **Testing**: 5-10 minutes

## Next Steps

1. **Setup**: Install UV and dependencies
2. **Configure**: Update OANDA credentials  
3. **Test**: Run quick test first
4. **Execute**: Run full analysis
5. **Analyze**: Review results and performance
6. **Deploy**: Implement live trading (optional)

The system will automatically handle the 2016-2024 training and 2024-2025 testing workflow as specified!