"""
Test Script for NFL Touchdown Prediction
Run this to test the entire pipeline
"""
from config import FEATURES, NUMERIC_FEATURES
from data_manager import NFLDataManager
from model import NFLTouchdownModel


def main():
    # Configuration
    season = 2025
    week = 6
    force_reload = False  # Set to True to force reload from API
    
    print("\n" + "="*60)
    print("NFL TOUCHDOWN PREDICTION - TEST SCRIPT")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading data...")
    data_manager = NFLDataManager(data_source_type="nflreadpy")
    result = data_manager.load_and_process_data(season, week, force_reload=force_reload)
    
    df = result['df']
    current_week = result['current_week']
    
    print(f"✅ Training data shape: {df.shape}")
    print(f"✅ Current week shape: {current_week.shape}")
    
    # Step 2: Train Model
    print("\n[STEP 2] Training model...")
    model = NFLTouchdownModel(season, week)
    
    if model.model_exists():
        print(f"⚠️  Model already exists at {model.model_path}")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training...")
        else:
            success, message = model.train(df, FEATURES, NUMERIC_FEATURES)
            print(f"{'✅' if success else '❌'} {message}")
    else:
        success, message = model.train(df, FEATURES, NUMERIC_FEATURES)
        print(f"{'✅' if success else '❌'} {message}")
    
    # Step 3: Make Predictions
    print("\n[STEP 3] Making predictions...")
    predictions = model.predict(current_week, FEATURES, NUMERIC_FEATURES)
    
    # Display top 10 predictions
    print("\n" + "="*60)
    print("TOP 10 TOUCHDOWN PREDICTIONS")
    print("="*60)
    print(predictions[['player_name', 'team', 'position', 'probability', 'played']].head(10).to_string(index=False))
    
    # Save predictions
    output_file = f'predictions_s{season}_w{week}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to {output_file}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()