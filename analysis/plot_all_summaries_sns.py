import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

def plot_all_summaries(
    summary_files: Dict[str, str],
    output_path: str,
    log_scale: bool = False,
    showfliers: bool = False
) -> None:
    """
    Load summary CSVs for multiple models and render comparative box plots.
    'all' and 'random' results are duplicated across every alpha value for display.

    Args:
        summary_files: Map of model name to CSV path.
        output_path: File path for the saved figure.
        log_scale: Whether to use log scale on the y-axis.
        showfliers: Whether to show outliers.
    """
    # 1. Load data
    all_dfs: List[pd.DataFrame] = []
    for model_name, path_str in summary_files.items():
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] File for {model_name} not found at '{path}'. Skipping.")
            continue
        df = pd.read_csv(path)
        df['model'] = model_name
        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No summary files could be loaded.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 2. Preprocess
    combined_df['overall_rmse'] = np.sqrt(combined_df['overall_mse'])
    dep_df = combined_df[combined_df['condition'].isin(['topk','bottomk'])]
    indep_df = combined_df[combined_df['condition'].isin(['random','all'])]
    alphas = sorted(dep_df['alpha'].dropna().unique())
    if alphas and not indep_df.empty:
        indep_rep = pd.concat([indep_df.assign(alpha=a) for a in alphas], ignore_index=True)
        final_df = pd.concat([dep_df, indep_rep], ignore_index=True)
    else:
        final_df = combined_df

    # 3. Category ordering
    model_order = ['SVR','RF','DGM']
    cond_order = ['all','topk','random','bottomk']
    final_df['model'] = pd.Categorical(final_df['model'], categories=model_order, ordered=True)
    final_df['condition'] = pd.Categorical(final_df['condition'], categories=cond_order, ordered=True)

    # 4. Plot
    sns.set_theme(style='whitegrid')
    g = sns.catplot(
        data=final_df,
        x='alpha',
        y='overall_rmse',
        hue='condition',
        col='model',
        kind='box',
        order=alphas,
        hue_order=cond_order,
        height=6,
        aspect=0.8,
        showfliers=showfliers,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.9), title=None)

    # Apply log scale if needed.
    if log_scale:
        for ax in g.axes.flatten():
            ax.set_yscale('log')

    g.figure.suptitle('Performance of Source Selection Strategies Across Models', y=1.03)
    g.set_axis_labels('Alpha', 'Overall RMSE')
    g.set_titles('Model: {col_name}')

    # 6. Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Comparison chart saved as '{output_path}'")
    plt.show()

if __name__ == '__main__':
    summary_files = {'DGM':'outputs/dgm_summary.csv','SVR':'outputs/svr_summary.csv','RF':'outputs/rf_summary.csv'}
    plot_all_summaries(summary_files, 'outputs/comparison_plot_unified.png', log_scale=False, showfliers=False)
