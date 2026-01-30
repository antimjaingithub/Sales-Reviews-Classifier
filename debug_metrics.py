
import pandas as pd
import numpy as np
import ast
import sys
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bodywash_classification'))

def calculate_metrics():
    file_path = "bodywash_classification/output/validation_results.xlsx"
    if not os.path.exists(file_path):
        print("File not found.")
        return

    df = pd.read_excel(file_path)
    
    # Parse string lists to actual lists
    # They might be stored as "['A', 'B']"
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    y_true_lists = df['True Factors'].apply(parse_list).tolist()
    y_pred_lists = df['Predicted Factors'].apply(parse_list).tolist()
    
    # Get all unique factors
    all_factors = sorted(list(set([item for sublist in y_true_lists for item in sublist] + 
                                [item for sublist in y_pred_lists for item in sublist])))
                                
    mlb = MultiLabelBinarizer(classes=all_factors)
    y_true = mlb.fit_transform(y_true_lists)
    y_pred = mlb.fit_transform(y_pred_lists)
    
    # Calculate Metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))
    hamming = hamming_loss(y_true, y_pred)
    
    print("VALIDATION METRICS RECALCULATED")
    print("="*30)
    print(f"Micro F1:    {micro_f1:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Hamming:     {hamming:.4f}")
    print("-" * 30)
    
    # Per Factor
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print("Factor Breakdown:")
    for i, factor in enumerate(all_factors):
        print(f"{factor:<25} F1: {f1[i]:.4f}  (P: {prec[i]:.2f}, R: {rec[i]:.2f})")

if __name__ == "__main__":
    calculate_metrics()
