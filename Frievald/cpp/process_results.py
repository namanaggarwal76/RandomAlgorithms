import pandas as pd
from pathlib import Path
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
RAW_ERROR_CSV = RESULTS_DIR / "cpp_error_raw.csv"
ERROR_CSV = RESULTS_DIR / "error.csv"

def process_error_results():
    if not RAW_ERROR_CSV.exists():
        print(f"Error: {RAW_ERROR_CSV} not found.")
        return

    df = pd.read_csv(RAW_ERROR_CSV)
    
    # Aggregate
    # df has: trial, n, error_mode, iterations_to_detect
    # We want to compute survival probability for k = 1..max_k
    
    records = []
    max_k = 20 # Or derive from data
    
    # Group by error_mode
    for mode, group in df.groupby("error_mode"):
        trials = len(group)
        detection_counts = group["iterations_to_detect"].tolist()
        
        for k in range(1, max_k + 1):
            # detected_by_k: count where iterations_to_detect <= k AND iterations_to_detect != -1
            # In C++, I used -1 for not detected.
            detected_by_k = sum(1 for d in detection_counts if d != -1 and d <= k)
            
            cumulative_detection_prob = detected_by_k / trials
            survival_prob = 1.0 - cumulative_detection_prob
            theoretical_survival = 0.5 ** k
            theoretical_cumulative = 1.0 - theoretical_survival
            
            records.append({
                "k": k,
                "n": group["n"].iloc[0], # Assuming n is constant per mode/run
                "trials": trials,
                "cumulative_detection_prob": cumulative_detection_prob,
                "survival_prob": survival_prob,
                "theoretical_survival": theoretical_survival,
                "theoretical_cumulative": theoretical_cumulative,
                "error_mode": mode,
                "seed": 42 # Hardcoded in C++
            })
            
    out_df = pd.DataFrame(records)
    out_df.to_csv(ERROR_CSV, index=False)
    print(f"Processed error results saved to {ERROR_CSV}")

if __name__ == "__main__":
    process_error_results()
