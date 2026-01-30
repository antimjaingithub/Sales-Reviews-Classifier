
import pandas as pd
import numpy as np
import ast
import sys
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bodywash_classification'))
from src.data_preprocessing import DataPreprocessor

def evaluate_test_set():
    print("Evaluating Test Set Predictions...")
    
    # 1. Load Ground Truth
    test_path = "Data/bodywash-test.xlsx"
    if not os.path.exists(test_path):
        print("Test data file not found.")
        return
        
    # Use DataPreprocessor to handle Long Format conversion if necessary
    # But wait, test data might be long format too? 
    # Let's check if we need to pivot.
    processor = DataPreprocessor(test_path)
    df_true_raw = processor.load_data()
    
    if 'Level 1 Factors' in df_true_raw.columns:
        # It's likely long format or explicit list.
        # Let's use the preprocess method to be sure we get the same wide format
        # IMPORTANT: We need to use is_test=False to trigger the Long-to-Wide conversion logic
        # strictly for evaluation purposes (even though it's the "test" file).
        gt_processor = DataPreprocessor(test_path, is_test=False) 
        df_true = gt_processor.preprocess()
        print(f"Ground Truth Loaded. Items: {len(df_true)}")
    else:
        print(f"Could not find 'Level 1 Factors' in {test_path}")
        return

    # 2. Load Predictions
    pred_path = "bodywash_classification/output/final_predictions.xlsx"
    if not os.path.exists(pred_path):
        print("Prediction file not found.")
        return
        
    df_pred_raw = pd.read_excel(pred_path)
    print(f"Predictions Loaded. Items: {len(df_pred_raw)}")
    
    # 3. Align Dataframes
    # Merge on 'Core Item' to ensure order matches
    merged = pd.merge(df_true, df_pred_raw[['Core Item', 'Level 1 Factors']], on='Core Item', suffixes=('_true', '_pred'))
    print(f"Aligned Items: {len(merged)}")
    
    print("\nDEBUG: Sample Data Comparison (First 5 rows)")
    print(merged[['Level 1 Factors_true', 'Level 1 Factors_pred']].head().to_string())
    print("-" * 30)

    # 4. Prepare Binarized Data
    factors = gt_processor.factors
    
    # Ground Truth Matrix
    # Check if _true is already binary or list?
    # DataPreprocessor with is_test=False returns a DATAFRAME with binary columns... NOT a 'Level 1 Factors' column with lists!
    # Ah! DataPreprocessor returns a dataframe where columns ARE the factors.
    
    print("DEBUG: Columns in Ground Truth DF:", df_true.columns.tolist())
    
    # If DataPreprocessor was used with Long-to-Wide, df_true has binary columns.
    # It does NOT have a 'Level 1 Factors' column anymore (it drops it or consumes it).
    # Wait, in DataPreprocessor:
    # "grouped = df_raw.groupby('Core Item')['Level 1 Factors'].apply(list).reset_index()" <-- This keeps 'Level 1 Factors' as a column of lists?
    # No, later "self.df = grouped".
    # And "grouped['cleaned_text'] = ..."
    # It seems 'Level 1 Factors' (the list column) IS preserved in `grouped`.
    
    if 'Level 1 Factors' in df_true.columns:
         y_true_lists = df_true['Level 1 Factors'].tolist()
    else:
        # If it's not there, maybe we need to reconstruct from binary columns?
        # But let's check the debug output first.
        pass

    y_true = merged[factors].values

    
    # Prediction Matrix
    # Predictions are strings "['A', 'B']" or "A, B"
    def parse_pred(x):
        if pd.isna(x): return []
        if isinstance(x, str):
            # Check if comma separated or list string
            if '[' in x:
                return ast.literal_eval(x)
            else:
                return [s.strip() for s in x.split(',') if s.strip()]
        return []

    pred_list = merged['Level 1 Factors_pred'].apply(parse_pred).tolist()
    
    mlb = MultiLabelBinarizer(classes=factors)
    y_pred = mlb.fit_transform(pred_list)
    
    # 5. Calculate Metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    exact_match = accuracy_score(y_true, y_pred) # Subset accuracy
    hamming = hamming_loss(y_true, y_pred)
    
    print("\n" + "#"*40)
    print("FINAL TEST SET RESULTS")
    print("#"*40)
    print(f"Micro F1 Score:      {micro_f1:.4f}")
    print(f"Macro F1 Score:      {macro_f1:.4f}")
    print(f"Exact Match Ratio:   {exact_match:.4f}")
    print(f"Hamming Loss:        {hamming:.4f}")
    print("-" * 30)
    
    # Per Factor
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    report_data = []
    print("\nFactor Specific Performance:")
    print(f"{'Factor':<25} | {'F1':<6} | {'Prec':<6} | {'Recall':<6}")
    print("-" * 50)
    for i, factor in enumerate(factors):
        print(f"{factor:<25} | {f1[i]:.4f} | {prec[i]:.4f} | {rec[i]:.4f}")
        report_data.append({
            'Factor': factor,
            'F1': f1[i],
            'Precision': prec[i],
            'Recall': rec[i]
        })
        
    return micro_f1, macro_f1, exact_match, report_data

if __name__ == "__main__":
    evaluate_test_set()
