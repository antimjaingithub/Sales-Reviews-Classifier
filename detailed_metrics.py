
import pandas as pd
import numpy as np
import ast
import sys
import os
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from scipy.stats import entropy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bodywash_classification'))
from src.data_preprocessing import DataPreprocessor

def calculate_detailed_metrics():
    print("CALCULATING DETAILED METRICS REPORT")
    print("="*40)
    
    # Paths
    VAL_RESULTS_PATH = "bodywash_classification/output/validation_results.xlsx"
    TEST_PREDS_PATH = "bodywash_classification/output/final_predictions_v3_cot.xlsx" # Updated to v3 CoT
    TRAIN_PATH = "Data/bodywash-train (1).xlsx"
    
    metrics_report = {}

    # ==========================================
    # 1. PRIMARY METRICS (Validation Set)
    # ==========================================
    print("\n1. PRIMARY METRICS (Validation Set)")
    
    if os.path.exists(VAL_RESULTS_PATH):
        df_val = pd.read_excel(VAL_RESULTS_PATH)
        
        # Parse Lists
        y_true_lists = df_val['True Factors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []).tolist()
        y_pred_lists = df_val['Predicted Factors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []).tolist()
        
        # Get Factors from Preprocessor to ensure consistence order
        processor = DataPreprocessor(TRAIN_PATH)
        all_factors = processor.factors
        
        mlb = MultiLabelBinarizer(classes=all_factors)
        y_true = mlb.fit_transform(y_true_lists)
        y_pred = mlb.fit_transform(y_pred_lists)
        
        # Metrics
        emr = np.mean(np.all(y_true == y_pred, axis=1))
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        hamming = hamming_loss(y_true, y_pred)
        
        metrics_report['Primary'] = {
            'EMR': emr,
            'Micro F1': micro_f1,
            'Macro F1': macro_f1,
            'Hamming': hamming
        }
        
        print(f"Exact Match Ratio (EMR): {emr*100:.2f}% (Target >60%)")
        print(f"Micro F1-Score:        {micro_f1:.4f} (Target >0.85)")
        print(f"Macro F1-Score:        {macro_f1:.4f} (Target >0.80)")
        print(f"Hamming Loss:          {hamming:.4f} (Target <0.15)")
        
        # Per Factor
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        print("\nPer-Factor F1:")
        for i, factor in enumerate(all_factors):
            print(f"  {factor:<25}: {f1_scores[i]:.4f}")
            
    else:
        print("Validation results file not found!")

    # ==========================================
    # 2. SECONDARY METRICS (Test Set Confidence)
    # ==========================================
    print("\n2. SECONDARY METRICS (Test Set Confidence)")
    
    if os.path.exists(TEST_PREDS_PATH):
        df_test = pd.read_excel(TEST_PREDS_PATH)
        
        # Check if score column exists
        if 'Confidence Scores' in df_test.columns:
            # Parse dict strings: "{'Fragrance': 1.0, 'Price': 0.5}"
            def get_avg_conf(score_str):
                try:
                    d = ast.literal_eval(score_str)
                    if not d: return 0.0
                    return np.mean(list(d.values()))
                except:
                    return 0.0
            
            df_test['avg_conf'] = df_test['Confidence Scores'].apply(get_avg_conf)
            
            avg_all_conf = df_test['avg_conf'].mean() # Mean of means
            conf_variance = df_test['avg_conf'].var()
            
            print(f"Average Confidence Score: {avg_all_conf:.4f} (Target >0.75)")
            print(f"Confidence Variance:      {conf_variance:.4f} (Target <0.1)")
        else:
            print("Confidence Scores column not found in predictions. Skipping confidence metrics.")
            
        # Ensemble Agreement (Simulated since we used single model)
        print("Ensemble Agreement Rate:  N/A (Single Model Used)")

    # ==========================================
    # 3. DISTRIBUTION CONSISTENCY
    # ==========================================
    print("\n3. DISTRIBUTION CONSISTENCY")
    
    # Train Distribution
    if os.path.exists(TRAIN_PATH):
        # We need to process train again to get label counts
        # (Assuming processor initialized above)
        df_train_raw = processor.load_data()
        
        # Check format
        train_counts = Counter()
        if 'Level 1 Factors' in df_train_raw.columns:
             for tags in df_train_raw['Level 1 Factors'].dropna():
                 # Handle string representation if needed, but likely just strings
                 # Wait, in raw excel it was Long format single strings?
                 # 'Level 1 Factors': 'Fragrance'
                 train_counts[str(tags).strip()] += 1
        
        # Normalize to probability distribution
        total_train_labels = sum(train_counts.values())
        train_dist = {k: v/total_train_labels for k,v in train_counts.items() if k in all_factors}
        
    # Test Distribution
    test_counts = Counter()
    # Parse predictions list "Brand Value, Price"
    for pred_str in df_test['Level 1 Factors'].dropna():
        if isinstance(pred_str, str):
            tags = [t.strip() for t in pred_str.split(',') if t.strip()]
            for t in tags:
                test_counts[t] += 1
                
    total_test_labels = sum(test_counts.values())
    test_dist = {k: v/total_test_labels for k,v in test_counts.items() if k in all_factors}
    
    # KL Divergence
    # Align distributions
    p = [] # Train
    q = [] # Test
    for f in all_factors:
        p.append(train_dist.get(f, 0.0001)) # Add small epsilon
        q.append(test_dist.get(f, 0.0001))
        
    kl_div = entropy(p, q)
    print(f"KL-Divergence:            {kl_div:.4f} (Target <0.1)")
    
    # Label Cardinality
    # Avg labels per item
    # Test
    def count_labels(s):
        if not isinstance(s, str): return 0
        return len([t for t in s.split(',') if t.strip()])
    
    df_test['label_count'] = df_test['Level 1 Factors'].apply(count_labels)
    test_cardinality = df_test['label_count'].mean()
    print(f"Label Cardinality (Test): {test_cardinality:.4f} (Target ~2.18)")
    
    # Factor Freq Match
    print("\nFactor Frequency Top 5 (Train vs Test):")
    sorted_train = sorted(train_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_test = sorted(test_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Train: {sorted_train}")
    print(f"Test:  {sorted_test}")

if __name__ == "__main__":
    calculate_detailed_metrics()
