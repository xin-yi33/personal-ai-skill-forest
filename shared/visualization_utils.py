"""
Visualization Utilities for Skill Forest Experiments
Provides consistent styling and helper functions for all experiment plots.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple

# Global style settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {
    'flat': '#4ECDC4',
    'single_tree': '#FF6B6B',
    'forest': '#45B7D1',
    'full_system': '#2ECC71',
    'no_M2': '#E74C3C',
    'no_M4': '#9B59B6',
    'no_M5': '#F39C12',
    'no_M6': '#1ABC9C',
    'no_M7': '#E67E22',
    'no_M9': '#3498DB',
    'primary': '#2C3E50',
    'secondary': '#7F8C8D',
    'accent': '#E74C3C',
    'success': '#2ECC71',
    'warning': '#F39C12',
}

METHOD_LABELS = {
    'flat': 'Flat ANN (FAISS)',
    'single_tree': 'Single B+ Tree',
    'forest': 'Skill Forest (Ours)',
}

ABLATION_LABELS = {
    'full_system': 'Full System',
    'no_M2': 'w/o M2 (Routing)',
    'no_M4': 'w/o M4 (Dependency)',
    'no_M5': 'w/o M5 (ABCD)',
    'no_M6': 'w/o M6 (Param Merge)',
    'no_M7': 'w/o M7 (Private Mask)',
    'no_M9': 'w/o M9 (Role Reduction)',
}


def setup_plot_style():
    """Set up consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


def save_figure(fig, save_path: str, filename: str):
    """Save figure to the specified path."""
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_bar_with_error(data: Dict[str, Dict[str, float]], metric_name: str,
                        title: str, ylabel: str, save_path: str, filename: str,
                        colors: Optional[Dict] = None, figsize: Tuple = (10, 6)):
    """Plot bar chart with error bars (mean ± std)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(data.keys())
    means = [data[m]['mean'] for m in methods]
    stds = [data[m]['std'] for m in methods]

    if colors is None:
        colors = COLORS

    bar_colors = [colors.get(m, '#95A5A6') for m in methods]
    labels = [METHOD_LABELS.get(m, ABLATION_LABELS.get(m, m)) for m in methods]

    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5,
                  color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return save_figure(fig, save_path, filename)


def plot_line_with_error(x_values: List, y_data: Dict[str, Dict[str, List]],
                         title: str, xlabel: str, ylabel: str,
                         save_path: str, filename: str,
                         colors: Optional[Dict] = None, figsize: Tuple = (10, 6)):
    """Plot line chart with error bands (mean ± std)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = COLORS

    for method, data in y_data.items():
        means = data['mean']
        stds = data['std']
        color = colors.get(method, '#95A5A6')
        label = METHOD_LABELS.get(method, ABLATION_LABELS.get(method, method))

        ax.plot(x_values, means, 'o-', color=color, label=label, linewidth=2, markersize=6)
        ax.fill_between(x_values,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return save_figure(fig, save_path, filename)


def plot_grouped_bar(data: Dict[str, Dict[str, float]], metrics: List[str],
                     title: str, ylabel: str, save_path: str, filename: str,
                     colors: Optional[Dict] = None, figsize: Tuple = (12, 6)):
    """Plot grouped bar chart comparing multiple methods across multiple metrics."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = COLORS

    methods = list(data.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        means = [data[method].get(m, {}).get('mean', 0) for m in metrics]
        stds = [data[method].get(m, {}).get('std', 0) for m in metrics]
        color = colors.get(method, '#95A5A6')
        label = METHOD_LABELS.get(method, ABLATION_LABELS.get(method, method))

        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                      color=color, label=label, alpha=0.9)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return save_figure(fig, save_path, filename)


def plot_heatmap(data: np.ndarray, row_labels: List[str], col_labels: List[str],
                 title: str, save_path: str, filename: str,
                 cmap: str = 'YlOrRd', figsize: Tuple = (10, 8)):
    """Plot heatmap."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=9)

    ax.set_title(title, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8)

    return save_figure(fig, save_path, filename)


def plot_dual_axis(x_values: List, y1_data: List, y2_data: List,
                   y1_label: str, y2_label: str, title: str,
                   save_path: str, filename: str, figsize: Tuple = (10, 6)):
    """Plot dual-axis chart."""
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=figsize)

    color1 = '#3498DB'
    color2 = '#E74C3C'

    ax1.bar(x_values, y1_data, alpha=0.6, color=color1, width=0.008, label=y1_label)
    ax1.set_xlabel('Threshold δ')
    ax1.set_ylabel(y1_label, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x_values, y2_data, 'o-', color=color2, linewidth=2, markersize=4, label=y2_label)
    ax2.set_ylabel(y2_label, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(title, fontweight='bold', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)

    ax1.grid(True, alpha=0.3, linestyle='--')

    return save_figure(fig, save_path, filename)


def plot_radar_chart(categories: List[str], values: Dict[str, List[float]],
                     title: str, save_path: str, filename: str,
                     colors: Optional[Dict] = None, figsize: Tuple = (8, 8)):
    """Plot radar/spider chart."""
    setup_plot_style()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    if colors is None:
        colors = COLORS

    for method, vals in values.items():
        vals_plot = vals + vals[:1]
        color = colors.get(method, '#95A5A6')
        label = METHOD_LABELS.get(method, ABLATION_LABELS.get(method, method))

        ax.plot(angles, vals_plot, 'o-', color=color, label=label, linewidth=2)
        ax.fill(angles, vals_plot, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    return save_figure(fig, save_path, filename)


def plot_token_comparison(theoretical: Dict[str, List[float]],
                          actual: Dict[str, Dict[str, float]],
                          n_values: List[int],
                          title: str, save_path: str, filename: str,
                          figsize: Tuple = (10, 6)):
    """Plot theoretical vs actual token consumption."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    for method, values in theoretical.items():
        color = COLORS.get(method, '#95A5A6')
        label = METHOD_LABELS.get(method, method)
        ax.plot(n_values, values, '--', color=color, label=f'{label} (Theoretical)', linewidth=2)

    for method, stats in actual.items():
        color = COLORS.get(method, '#95A5A6')
        label = METHOD_LABELS.get(method, method)
        ax.axhline(y=stats['mean'], color=color, linestyle='-', linewidth=2, alpha=0.7)
        ax.axhspan(stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                   alpha=0.1, color=color)
        ax.annotate(f'{label}: {stats["mean"]:.0f}±{stats["std"]:.0f}',
                    xy=(n_values[-1], stats['mean']), fontsize=9,
                    xytext=(10, 0), textcoords='offset points')

    ax.set_xlabel('Number of Skills (N)')
    ax.set_ylabel('Token Consumption')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return save_figure(fig, save_path, filename)


def create_results_table(data: Dict, columns: List[str], title: str,
                         save_path: str, filename: str):
    """Create and save a formatted results table as image."""
    setup_plot_style()

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(max(12, len(columns) * 1.5), max(4, len(df) * 0.5 + 1)))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')

    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    return save_figure(fig, save_path, filename)


def generate_significance_table(results: Dict, comparisons: List[Tuple[str, str]],
                                 metrics: List[str], save_path: str, filename: str):
    """Generate a significance test results table."""
    from scipy import stats as sp_stats

    rows = []
    for m1, m2 in comparisons:
        row = {'Comparison': f'{ABLATION_LABELS.get(m1, m1)} vs {ABLATION_LABELS.get(m2, m2)}'}
        for metric in metrics:
            vals1 = results[m1].get(metric, {}).get('values', [results[m1][metric]['mean']])
            vals2 = results[m2].get(metric, {}).get('values', [results[m2][metric]['mean']])

            if len(vals1) >= 2 and len(vals2) >= 2:
                t_stat, p_val = sp_stats.ttest_rel(vals1[:5], vals2[:5])
                diff = np.mean(vals1) - np.mean(vals2)
                pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2)
                cohens_d = diff / pooled_std if pooled_std > 0 else 0

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                row[metric] = f'{sig} (p={p_val:.4f}, d={cohens_d:.2f})'
            else:
                row[metric] = 'N/A'
        rows.append(row)

    df = pd.DataFrame(rows)
    create_results_table(df, df.columns.tolist(), 'Statistical Significance Tests',
                         save_path, filename)
