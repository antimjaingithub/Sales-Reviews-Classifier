
import os
import sys
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bodywash_classification'))

from src.data_preprocessing import DataPreprocessor
from src.prompt_engineering import PromptManager
from src.api_client import GroqClient
from src.ensemble_predictor import EnsemblePredictor

def main():
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: API Key missing")
        return

    # Initialize Components
    client = GroqClient()
    pm = PromptManager()
    # Configure 3-Judge Ensemble (Prompt Diversity)
    predictor = EnsemblePredictor(client, pm)
    predictor.models_config = [
        {'name': 'llama-3.1-8b-instant', 'temp': 0.1, 'weight': 1.0, 'variant': 'strict_cot'},
        {'name': 'llama-3.1-8b-instant', 'temp': 0.1, 'weight': 1.0, 'variant': 'reasoning'},
        {'name': 'llama-3.1-8b-instant', 'temp': 0.1, 'weight': 1.0, 'variant': 'reflection'}
    ]
    
    # 2. Load ONLY Test Data
    TEST_PATH = "Data/bodywash-test.xlsx"
    if not os.path.exists(TEST_PATH):
        print(f"Test file not found: {TEST_PATH}")
        return

    test_processor = DataPreprocessor(TEST_PATH, is_test=True)
    df_test = test_processor.preprocess()
    
    print(f"Predicting on {len(df_test)} test items...")
    
    # Check for existing partial results to resume? 
    # For now, just run fresh or overwrite.
    
    results = []
    
    for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
        item_text = row.get('Core Item')
        if not item_text:
            continue
            
        factors, scores = predictor.predict_item(item_text)
        
        # Log empty predictions to console so we know if it failed
        if not factors:
            print(f"Warning: Empty prediction for Item {index}")
        
        results.append({
            'Core Item': item_text,
            'Level 1 Factors': ', '.join(factors)
        })
        
    # Save Final Output
    output_df = pd.DataFrame(results)
    os.makedirs('bodywash_classification/output', exist_ok=True)
    output_path = 'bodywash_classification/output/final_predictions_v4_ensemble.xlsx'
    output_df.to_excel(output_path, index=False)
    
    print(f"\nSUCCESS. Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
