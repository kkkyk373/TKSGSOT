import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from typing import Dict, List


# ===== Shared utilities =====
def _prepare_alpha_broadcast(df: pd.DataFrame) -> tuple[pd.DataFrame, List[float]]:
    """
    Duplicate 'random'/'all' rows over the alpha axis for topk/bottomk visualizations.
    """
    alpha_dependent = df[df['condition'].isin(['topk', 'bottomk'])]
    alpha_independent = df[df['condition'].isin(['random', 'all'])]

    # Use alpha values from topk/bottomk configurations if available; otherwise fall back.
    alphas = sorted(alpha_dependent['alpha'].dropna().unique().tolist())
    if not alphas:
        alphas = sorted(df['alpha'].dropna().unique().tolist())

    if alphas and not alpha_independent.empty:
        replicated = pd.concat(
            [alpha_independent.assign(alpha=a) for a in alphas],
            ignore_index=True
        )
        final_df = pd.concat([alpha_dependent, replicated], ignore_index=True)
    else:
        # Inject a dummy alpha when the column is entirely missing.
        if not alphas:
            alphas = [0.0]
            final_df = df.copy()
            final_df['alpha'] = alphas[0]
        else:
            final_df = df.copy()

    return final_df, alphas


def _plot_box(ax, df_model, alphas, cond_order, color_map, showfliers, ylabel=None, title=None, log_scale=False):
    """
    Shared helper to render each subplot; df_model must contain 'overall_rmse'.
    """
    n_cond = len(cond_order)
    box_data, colors, positions = [], [], []

    for i, a in enumerate(alphas):
        base = i * (n_cond + 1)
        for j, cond in enumerate(cond_order):
            vals = df_model.loc[
                (df_model['alpha'] == a) &
                (df_model['condition'] == cond),
                'overall_rmse'
            ].values
            box_data.append(vals if vals.size else [np.nan])
            colors.append(color_map.get(cond, '#cccccc'))
            positions.append(base + j)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=showfliers,
        whis=1.5,
        medianprops=dict(color='black', linewidth=2)
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)

    centers = [i*(n_cond+1) + (n_cond-1)/2 for i in range(len(alphas))]
    if centers:
        ax.set_xticks(centers)
        xticklabels = []
        for a in alphas:
            if np.isfinite(a) and a != 0.0:
                xticklabels.append(f"α={int(a) if float(a).is_integer() else a}")
            else:
                xticklabels.append("α=–")
        ax.set_xticklabels(xticklabels, rotation=0)

    ax.set_xlabel("Alpha")
    if log_scale:
        ax.set_yscale('log')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def _add_legend(fig, color_map, cond_order):
    label_map = {
        'all':     'all',
        'topk':    'topk',
        'random':  'random',
        'bottomk': 'bottomk'
    }
    handles = [Patch(color=color_map[c], label=label_map[c]) for c in cond_order]
    fig.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(0.1, 0.9),
        borderaxespad=0.,
        title_fontsize=18,
        fontsize=18,
    )


# ===== Pair-level plots =====
def plot_pair_summaries(
    summary_files: Dict[str, str],
    output_path: str,
    log_scale: bool = False,
    showfliers: bool = False
) -> None:
    """Load pair summaries (overall_mse) and visualize overall_rmse = sqrt(overall_mse)."""
    # 1. Load
    dfs: List[pd.DataFrame] = []
    for model, path_str in summary_files.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] '{model}' file not found: {p}")
            continue
        df = pd.read_csv(p)
        df['model'] = model
        if 'condition' not in df.columns:
            df['condition'] = np.nan
        if 'alpha' not in df.columns:
            df['alpha'] = np.nan
        df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')

        if 'overall_mse' not in df.columns:
            print(f"[WARN] Skip '{model}': overall_mse not found.")
            continue

        df['overall_rmse'] = np.sqrt(df['overall_mse'])
        dfs.append(df)

    if not dfs:
        print("[ERROR] No summary files loaded. Aborting.")
        return

    data = pd.concat(dfs, ignore_index=True)

    # 2. Broadcast alpha-independent rows.
    final_df, alphas = _prepare_alpha_broadcast(data)

    # 3. 設定
    desired_order = ['SVR', 'RF', 'DGM']
    present_models = [m for m in desired_order if m in final_df['model'].unique().tolist()]
    if not present_models:
        present_models = sorted(final_df['model'].unique().tolist())

    cond_order = ['all', 'topk', 'random', 'bottomk']
    color_map = {
        'all':     '#1f77b4',
        'topk':    '#ff7f0e',
        'random':  '#2ca02c',
        'bottomk': '#d62728',
    }

    # 4. 描画
    n_models = len(present_models)
    fig, axes = plt.subplots(1, n_models, sharey=True, figsize=(5*n_models, 6))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, present_models):
        df_model = final_df[final_df['model'] == model]
        _plot_box(
            ax=ax,
            df_model=df_model,
            alphas=alphas,
            cond_order=cond_order,
            color_map=color_map,
            showfliers=showfliers,
            ylabel="Overall RMSE" if ax is axes[0] else None,
            title=model,
            log_scale=log_scale
        )

    # 5. 凡例
    _add_legend(fig, color_map, cond_order)

    # 6. 統計出力
    print("\n[Pair] Mean & Std of RMSE by model, condition, alpha:")
    stats = final_df.groupby(['model', 'condition', 'alpha'])['overall_rmse'].agg(['mean', 'std', 'count']).round(3).reset_index()
    print(stats.to_string(index=False))

    # 7. 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(output_path).with_suffix('.svg')), dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved to {output_path}")
    plt.show()


# ===== Graph-level plots =====
def plot_graph_summaries(
    summary_files: Dict[str, str],
    output_path: str,
    log_scale: bool = False,
    showfliers: bool = False
) -> None:
    """Load graph summaries and visualize rmse_mean (or sqrt(mse_mean))."""
    # 1. Load
    dfs: List[pd.DataFrame] = []
    for model, path_str in summary_files.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] '{model}' file not found: {p}")
            continue
        df = pd.read_csv(p)
        df['model'] = model
        if 'condition' not in df.columns:
            df['condition'] = np.nan
        if 'alpha' not in df.columns:
            df['alpha'] = np.nan
        df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')

        # Harmonize metrics: prefer rmse_mean, otherwise fallback on sqrt(mse_mean).
        if 'rmse_mean' in df.columns and df['rmse_mean'].notna().any():
            df['overall_rmse'] = df['rmse_mean']
        elif 'mse_mean' in df.columns and df['mse_mean'].notna().any():
            df['overall_rmse'] = np.sqrt(df['mse_mean'])
        else:
            print(f"[WARN] Skip '{model}': neither rmse_mean nor mse_mean found.")
            continue

        dfs.append(df)

    if not dfs:
        print("[ERROR] No graph summary files loaded. Aborting.")
        return

    data = pd.concat(dfs, ignore_index=True)

    # 2. Broadcast alpha-independent rows.
    final_df, alphas = _prepare_alpha_broadcast(data)

    # 3. 設定
    desired_order = ['SVR', 'RF', 'DGM']
    present_models = [m for m in desired_order if m in final_df['model'].unique().tolist()]
    if not present_models:
        present_models = sorted(final_df['model'].unique().tolist())

    cond_order = ['all', 'topk', 'random', 'bottomk']
    color_map = {
        'all':     '#1f77b4',
        'topk':    '#ff7f0e',
        'random':  '#2ca02c',
        'bottomk': '#d62728',
    }

    # 4. 描画
    n_models = len(present_models)
    fig, axes = plt.subplots(1, n_models, sharey=True, figsize=(5*n_models, 6))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, present_models):
        df_model = final_df[final_df['model'] == model]
        _plot_box(
            ax=ax,
            df_model=df_model,
            alphas=alphas,
            cond_order=cond_order,
            color_map=color_map,
            showfliers=showfliers,
            ylabel="RMSE (mean over graphs)" if ax is axes[0] else None,
            title=model,
            log_scale=log_scale
        )

    # 5. 凡例
    _add_legend(fig, color_map, cond_order)

    # 6. 統計出力
    print("\n[Graph] Mean & Std of RMSE (rmse_mean or sqrt(mse_mean)) by model, condition, alpha:")
    stats = final_df.groupby(['model', 'condition', 'alpha'])['overall_rmse'].agg(['mean', 'std', 'count']).round(3).reset_index()
    print(stats.to_string(index=False))

    # 7. 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(output_path).with_suffix('.svg')), dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved to {output_path}")
    plt.show()


# ===== 実行例 =====
if __name__ == '__main__':
    # Graph summaries
    summary_files = {
        'DGM': 'outputs/dgm_graph_summary.csv',
        'SVR': 'outputs/svr_graph_summary.csv',
        'RF':  'outputs/rf_graph_summary.csv'
    }
    plot_graph_summaries(summary_files, 'outputs/comparison_graph_plot_lier.png', log_scale=True, showfliers=True)
    
    # Pair summaries
    summary_files = {
        'DGM': 'outputs/dgm_pair_summary.csv',
        'SVR': 'outputs/svr_pair_summary.csv',
        'RF':  'outputs/rf_pair_summary.csv'
    }
    plot_pair_summaries(summary_files, 'outputs/comparison_pair_plot_lier.png', log_scale=True, showfliers=True)
