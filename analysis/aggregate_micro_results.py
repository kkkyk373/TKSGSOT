"""
Aggregate per-pair results and compute an overall MSE (over_mse) weighted by sample counts.
The weighting uses the total number of OD pairs instead of the number of graphs.

Use case: treat runs with different seeds as random variables and summarize their mean and std.
"""
import os
import glob
import json
import pandas as pd
import numpy as np

def aggregate_results_from_json(
    model_name="svr",
    input_root_dir="results/model/options",
    output_csv_path="outputs/model_pair_summary.csv"
):
    """Aggregate every JSON file under input_root_dir and dump a micro-level MSE summary."""

    print(f"--- Starting aggregation for {model_name.upper()} from '{input_root_dir}' ---")

    param_keys = [
        'condition', 'alpha', 'seed', 'top_k', 'bottom_k', 
        'max_samples', 'epochs', 'batch_size', 'lr'
    ]

    # --- Step 1: locate JSON files ---
    search_pattern = os.path.join(input_root_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"[ERROR] No result JSON files found in '{input_root_dir}'.")
        return

    all_runs_data = []

    # --- Step 2: process every JSON file and compute the weighted mean ---
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Retrieve metadata and per-target results.
        metadata = data.get("metadata", {})
        results = data.get("results", [])

        if not metadata or not results:
            print(f"[WARN] Invalid JSON structure in {filepath}. Skipping.")
            continue

        total_mse_product = 0
        total_test_samples = 0
        
        for r in results:
            mse = r.get("mse")
            test_samples = r.get("test_samples")
            if mse is not None and test_samples is not None and test_samples > 0:
                total_mse_product += mse * test_samples
                total_test_samples += test_samples

        if total_test_samples > 0:
            overall_mse = total_mse_product / total_test_samples
        else:
            overall_mse = np.nan

        run_data = {key: metadata.get(key) for key in param_keys}
        run_data['overall_mse'] = overall_mse

        # Alpha is irrelevant for 'random' and 'all', so normalize it to NaN.
        if run_data.get('condition') in ['random', 'all']:
            run_data['alpha'] = np.nan

        all_runs_data.append(run_data)

    if not all_runs_data:
        print(f"[ERROR] No data could be aggregated from '{input_root_dir}'.")
        return

    # --- Step 3: convert to DataFrame and save ---
    df = pd.DataFrame(all_runs_data)

    # --- Step 4: drop duplicates ---
    df.drop_duplicates(subset=param_keys, keep='last', inplace=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"[INFO] Aggregation complete. Results are saved in '{output_csv_path}'")
    print(f"--- Summary for {model_name.upper()} ---")
    print(df.head())


if __name__ == '__main__':
    # SVR (raw)
    aggregate_results_from_json(
        model_name="svr_raw",
        input_root_dir="results/svr/raw",
        output_csv_path="outputs/svr_pair_summary.csv"
    )
    
    # RF (raw)
    aggregate_results_from_json(
        model_name="rf_raw",
        input_root_dir="results/rf/raw",
        output_csv_path="outputs/rf_pair_summary.csv"
    ) 

    aggregate_results_from_json(
        model_name="dgm",
        input_root_dir="results/dgm/raw",
        output_csv_path="outputs/dgm_pair_summary.csv"
    )
