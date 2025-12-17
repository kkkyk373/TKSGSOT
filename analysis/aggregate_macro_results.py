"""
Aggregate every experiment at the graph level and output a DataFrame with summary statistics.
Example row:
  condition  alpha  seed  top_k  bottom_k  max_samples epochs batch_size    lr      mse_mean       mse_std   rmse_mean   rmse_std  n_targets_used
0   bottomk  100.0     3    100       100         5000   None       None  None  27775.570449  43900.157036  137.552714  94.308017             227
The evaluation set currently contains 227 graphs.
"""

import os
import glob
import json
import pandas as pd
import numpy as np


# ===== Common utilities =====
def _finite(values):
    """Keep only finite numeric values."""
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean_std_sample(arr: np.ndarray):
    """Return the mean and sample standard deviation (ddof=1)."""
    if arr.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan
    return mean, std


# ===== Aggregation entry point =====
def aggregate_graph_results_from_json(
    model_name: str,
    input_root_dir: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """Aggregate per-target MSE/RMSE metrics and save the graph-level summary to CSV."""
    print(f"--- Starting graph-level aggregation for {model_name.upper()} from '{input_root_dir}' ---")

    # Metadata keys to preserve.
    param_keys = [
        'condition', 'alpha', 'seed', 'top_k', 'bottom_k',
        'max_samples', 'epochs', 'batch_size', 'lr'
    ]

    # Find JSON files.
    search_pattern = os.path.join(input_root_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"[ERROR] No result JSON files found in '{input_root_dir}'.")
        return pd.DataFrame(columns=param_keys + [
            'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std', 'n_targets_used'
        ])

    all_rows = []

    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        results = data.get("results", [])

        if not metadata or not isinstance(results, list):
            print(f"[WARN] Invalid JSON structure in {filepath}. Skipping.")
            continue

        # 1) Extract per-target MSE values, dropping NaNs.
        mse_list = _finite([r.get('mse', np.nan) for r in results])

        # 2) Compute RMSE = sqrt(MSE) per target (ignore negative values).
        rmse_list = _finite([np.sqrt(x) for x in mse_list if np.isfinite(x) and x >= 0.0])

        # 3) Compute summary statistics.
        mse_mean, mse_std = _mean_std_sample(mse_list)
        rmse_mean, rmse_std = _mean_std_sample(rmse_list)

        # 4) Build the output row.
        row = {key: metadata.get(key) for key in param_keys}
        row['mse_mean'] = mse_mean
        row['mse_std'] = mse_std
        row['rmse_mean'] = rmse_mean
        row['rmse_std'] = rmse_std
        row['n_targets_used'] = int(mse_list.size)

        # Random/all conditions do not depend on alpha.
        if row.get('condition') in ['random', 'all']:
            row['alpha'] = np.nan

        all_rows.append(row)

    # Convert to DataFrame.
    df = pd.DataFrame(all_rows)

    # Deduplicate rows.
    df.drop_duplicates(subset=param_keys, keep='last', inplace=True)

    # Sort for readability.
    df.sort_values(by=['condition', 'alpha', 'seed'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save summary.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"[INFO] Aggregation complete. Results saved to '{output_csv_path}'")
    print(f"--- Summary [{model_name.upper()}] ---")
    print(df.head())

    return df


if __name__ == '__main__':
    # Configure the models to aggregate.
    models = ['svr', 'rf', 'dgm']

    for model in models:
        in_dir = os.path.join("results", model, "raw")
        out_csv = os.path.join("outputs", f"{model}_graph_summary.csv")
        aggregate_graph_results_from_json(model_name=model, input_root_dir=in_dir, output_csv_path=out_csv)
