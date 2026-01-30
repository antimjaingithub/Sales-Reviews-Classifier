
import pandas as pd
import os
import sys

# Paths
VAL_RESULTS_PATH = "bodywash_classification/output/validation_results.xlsx"
TEST_PREDS_PATH = "bodywash_classification/output/final_predictions_v3_cot.xlsx" # Using v3 CoT
OUTPUT_PATH = "bodywash_classification/output/BodyWash_Project_Deliverable.xlsx"

def generate_report():
    print("Generating Final Deliverable Excel...")
    
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        
        # Sheet 1: Final Predictions
        if os.path.exists(TEST_PREDS_PATH):
            df_test = pd.read_excel(TEST_PREDS_PATH)
            df_test.to_excel(writer, sheet_name='Final Predictions (Optimized)', index=False)
            print("Added Test Predictions Sheet")
        else:
            # Fallback to v1 if v2 not ready
            if os.path.exists("bodywash_classification/output/final_predictions.xlsx"):
                 df_test = pd.read_excel("bodywash_classification/output/final_predictions.xlsx")
                 df_test.to_excel(writer, sheet_name='Final Test Predictions (v1)', index=False)
                 print("Added Test Predictions Sheet (fallback v1)")

        # Sheet 2: Validation Analysis
        if os.path.exists(VAL_RESULTS_PATH):
            df_val = pd.read_excel(VAL_RESULTS_PATH)
            df_val.to_excel(writer, sheet_name='Validation Analysis', index=False)
            print("Added Validation Analysis Sheet")
            
        # Sheet 3: Metadata / Definition
        meta_data = [
            {"Info": "Model", "Value": "Llama 3.3 70B Versatile"},
            {"Info": "Metric: Validation F1", "Value": "0.45"},
            {"Info": "Metric: Validation Precision", "Value": "High (Fragrance >90%)"},
            {"Info": "Metric: Validation Recall", "Value": "Low (Conservative)"},
        ]
        pd.DataFrame(meta_data).to_excel(writer, sheet_name='Project Metadata', index=False)
        
    print(f"Success! Final deliverable saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_report()
