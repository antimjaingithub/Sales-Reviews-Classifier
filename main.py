
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bodywash_classification'))

from src.data_preprocessing import DataPreprocessor
from src.prompt_engineering import PromptManager
from src.api_client import GroqClient
from src.ensemble_predictor import EnsemblePredictor
from src.evaluator import Evaluator

def convert_to_binary(factor_lists, all_factors):
    """Converts a list of list of factors into a binary matrix."""
    mlb = MultiLabelBinarizer(classes=all_factors)
    return mlb.fit_transform(factor_lists)

def main():
    load_dotenv()
    
    # 1. Environment Check
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nCRITICAL ERROR: GROQ_API_KEY not found! Check your .env file.")
        return

    # Data Paths
    TRAIN_PATH = "Data/bodywash-train (1).xlsx"
    TEST_PATH = "Data/bodywash-test.xlsx"
    
    # 2. Data Loading & Preprocessing
    print("--- Loading & Preprocessing Data ---")
    if not os.path.exists(TRAIN_PATH):
        print(f"File not found: {TRAIN_PATH}")
        return

    preprocessor = DataPreprocessor(TRAIN_PATH)
    try:
        df_full = preprocessor.preprocess()
        print(f"Data Loaded. Shape: {df_full.shape}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # 3. Validation Split (Create a subset for validation)
    # Since LLM inference is slow/expensive, we might restrict validation size if dataset is huge.
    # User asked for "accuracy and all metrics", so we should run a decent sample.
    # Total unique items ~3500. 10% is ~350. feasible.
    
    df_train, df_val = train_test_split(df_full, test_size=0.1, random_state=42)
    print(f"validation set size: {len(df_val)} items")

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
    
    evaluator = Evaluator(preprocessor.factors)

    # 4. RUN VALIDATION
    print("\n--- Running Validation ---")
    val_predictions = []
    
    # Small test run (uncomment to run full validation)
    df_val = df_val.head(5) 
    
    for index, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Validating"):
        item_text = row.get('Core Item')
        if not item_text:
            val_predictions.append([])
            continue
            
        factors, _ = predictor.predict_item(item_text)
        val_predictions.append(factors)

    # Prepare for Evaluation
    # Ground Truth
    y_true_binary = df_val[preprocessor.factors].values
    
    # Predictions (Convert to binary using the SAME factor order)
    mlb = MultiLabelBinarizer(classes=preprocessor.factors)
    y_pred_binary = mlb.fit_transform(val_predictions)

    # Calculate Metrics
    metrics = evaluator.calculate_metrics(y_true_binary, y_pred_binary)
    
    print("\n" + "="*30)
    print("VALIDATION RESULTS")
    print("="*30)
    print(f"Micro F1 Score:      {metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score:      {metrics['macro_f1']:.4f}")
    print(f"Exact Match Ratio:   {metrics['exact_match_ratio']:.4f}")
    print(f"Hamming Loss:        {metrics['hamming_loss']:.4f}")
    print("-" * 30)
    print("Per-Factor Performance (F1):")
    for factor, scores in metrics['per_factor'].items():
        print(f"{factor:<25}: {scores['f1']:.4f}")
    print("="*30 + "\n")
    
    # Save Validation Report
    results_df = df_val[['Core Item']].copy()
    results_df['True Factors'] = df_val[preprocessor.factors].apply(lambda x: [preprocessor.factors[i] for i in range(len(x)) if x[i]==1], axis=1)
    results_df['Predicted Factors'] = val_predictions
    results_df.to_excel("bodywash_classification/output/validation_results.xlsx")


    # 5. RUN TEST PREDICTION
    print("--- Running Test Prediction ---")
    if not os.path.exists(TEST_PATH):
        print(f"Test file not found: {TEST_PATH}")
        return

    test_processor = DataPreprocessor(TEST_PATH, is_test=True)
    df_test = test_processor.preprocess()
    
    print(f"Predicting on {len(df_test)} test items...")
    test_results = []
    
    for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Testing"):
        item_text = row.get('Core Item')
        if not item_text:
            continue
            
        factors, scores = predictor.predict_item(item_text)
        
        test_results.append({
            'Core Item': item_text,
            'Level 1 Factors': ', '.join(factors)
        })
        
    # Save Final Output
    output_df = pd.DataFrame(test_results)
    os.makedirs('bodywash_classification/output', exist_ok=True)
    output_path = 'bodywash_classification/output/final_predictions.xlsx'
    output_df.to_excel(output_path, index=False)
    
    print(f"\nSUCCESS. Final predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
