"""
Diagnostic script to check what's happening
"""
from data_manager import NFLDataManager
import os
from config import HISTORICAL_DATA_FILE

print("="*60)
print("NFL DATA DIAGNOSTIC")
print("="*60)

# Check if cache exists
print(f"\n1. Checking cache file: {HISTORICAL_DATA_FILE}")
if os.path.exists(HISTORICAL_DATA_FILE):
    print("   ✅ Cache file exists")
    import pandas as pd
    cached = pd.read_parquet(HISTORICAL_DATA_FILE)
    print(f"   Total records: {len(cached)}")
    print(f"   Seasons: {sorted(cached['season'].unique())}")
    print(f"   Weeks per season:")
    for season in sorted(cached['season'].unique()):
        weeks = sorted(cached[cached['season'] == season]['week'].unique())
        print(f"      {season}: weeks {min(weeks)}-{max(weeks)}")
else:
    print("   ❌ Cache file does NOT exist - will need fresh load")

# Try loading data
print("\n2. Testing data load for 2024, Week 10...")
try:
    dm = NFLDataManager()
    result = dm.load_and_process_data(2024, 10, force_reload=True)
    
    df = result['df']
    current_week = result['current_week']
    
    print(f"   ✅ Success!")
    print(f"   Training data: {len(df)} records")
    print(f"   Current week: {len(current_week)} players")
    
    if len(df) > 0:
        print(f"   Training data seasons: {sorted(df['season'].unique())}")
        print(f"   Training data has 'played' column: {'played' in df.columns}")
        if 'played' in df.columns:
            print(f"   Played==1 records: {len(df[df['played']==1])}")
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)