import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Tuple, Union

# def plot_ycol_vs_jobs(y_column, df):
#     """
#     Plots a scatter plot and regression line between 'number_of_jobs' and the specified y_column.
#     Also prints the Pearson correlation coefficient and p-value.

#     Parameters:
#     - y_column (str): The name of the column to use for the y-axis.
#     - df (pd.DataFrame): The DataFrame containing the data.
#     """
#     sns.set_theme(style="whitegrid")

#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(data=df, x='number_of_jobs', y=y_column, s=100)
#     #sns.regplot(data=df, x='number_of_jobs', y=y_column, scatter=True, ci=None)

#     plt.xlabel('Number of Jobs')
#     plt.ylabel(y_column.replace('_', ' ').title())
#     plt.title(f'Correlation between Number of Jobs and {y_column.replace("_", " ").title()}')

#     plt.show()

#     corr_coef, p_value = pearsonr(df['number_of_jobs'], df[y_column])
#     print(f"Pearson correlation coefficient: {corr_coef:.3f}")
#     print(f"P-value: {p_value:.3e}")

def plot_ycol_vs_jobs(y_column: str,
                      df: pd.DataFrame,
                      threshold_ms: Optional[float] = None,
                      threshold_label: str = "threshold",
                      figsize: Tuple[int,int] = (14, 5),
                      show_regression: bool = True,
                      point_size: int = 60
                      ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]]:
    """
    Plot number_of_jobs vs y_column. If threshold_ms is provided, produce two side-by-side panels:
      - left: full data (own y axis)
      - right: truncated data (y < threshold_ms) (own y axis)
    If threshold_ms is None, produce a single plot.

    Returns (fig, ax) for single plot or (fig, (ax_left, ax_right)) for split plot.
    """
    if 'number_of_jobs' not in df.columns:
        raise ValueError("DataFrame must contain 'number_of_jobs' column")
    if y_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{y_column}' column")

    sns.set_theme(style="whitegrid")
    df_clean = df.dropna(subset=['number_of_jobs', y_column]).copy()

    def _plot_panel(ax, data, title):
        sns.scatterplot(data=data, x='number_of_jobs', y=y_column, s=point_size, ax=ax)
        ax.set_xlabel('Number of Jobs')
        ax.set_ylabel(y_column.replace('_', ' ').title())
        ax.set_title(title)

        if show_regression and len(data) >= 2:
            x = data['number_of_jobs'].astype(float).values
            y = data[y_column].astype(float).values
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            x_pred = np.linspace(x.min(), x.max(), 200)
            X_pred = sm.add_constant(x_pred)
            y_pred = model.predict(X_pred)
            ax.plot(x_pred, y_pred, color='red', lw=1.5, label='OLS fit')

            try:
                r, p = pearsonr(x, y)
                stats_text = f"r={r:.3f}, p={p:.2e}"
            except Exception:
                stats_text = "r=NA, p=NA"

            ax.text(0.98, 0.02, stats_text,
                    transform=ax.transAxes,
                    ha='right', va='bottom',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.65, edgecolor='none'))
        else:
            ax.text(0.98, 0.02, "insufficient data for regression",
                    transform=ax.transAxes,
                    ha='right', va='bottom',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.65, edgecolor='none'))

        ax.legend(loc='upper left')

    # Single plot (no threshold)
    if threshold_ms is None:
        fig, ax = plt.subplots(figsize=figsize)
        _plot_panel(ax, df_clean, f"Number of Jobs vs {y_column.replace('_', ' ').title()} (all data)")
        plt.tight_layout()
        return fig, ax

    # Two-panel plot with independent y axes
    df_below = df_clean[df_clean[y_column] < threshold_ms].copy()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    _plot_panel(ax_left, df_clean, f"All data (n={len(df_clean)})")
    _plot_panel(ax_right, df_below, f"{y_column.replace('_', ' ').title()} < {threshold_ms} ({threshold_label}) (n={len(df_below)})")

    # Set independent y-limits (auto by Matplotlib). Optionally pad a little for each axis:
    for ax, data in ((ax_left, df_clean), (ax_right, df_below)):
        if len(data) > 0:
            y_min, y_max = data[y_column].min(), data[y_column].max()
            if y_min == y_max:
                # small expand when constant value
                pad = max(1.0, abs(y_min) * 0.05)
                ax.set_ylim(y_min - pad, y_max + pad)
            else:
                pad = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

    plt.tight_layout()
    return fig, (ax_left, ax_right)


def plot_phase_cumulative(df: pd.DataFrame,
                          phase_col: str = 'Phase',
                          share_col: str = 'share_of_total_time',
                          exclude_total: bool = True,
                          cmap: str = 'tab10',
                          figsize: Tuple[int, int] = (10, 4.5),
                          label_pct: bool = True,
                          alpha: float = 0.45) -> Tuple[plt.Figure, plt.Axes]:
    if phase_col not in df.columns or share_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{phase_col}' and '{share_col}'")

    plot_df = df.copy()
    if exclude_total:
        plot_df = plot_df[plot_df[phase_col].astype(str).str.lower() != 'total'].reset_index(drop=True)

    plot_df[share_col] = pd.to_numeric(plot_df[share_col], errors='coerce').fillna(0)
    total_share = plot_df[share_col].sum()
    if total_share <= 0:
        raise ValueError("Sum of share_col must be positive")

    plot_df['share_norm_pct'] = plot_df[share_col] / total_share * 100
    cum = plot_df['share_norm_pct'].cumsum().values
    # build cumulative curve points (include 0 and 100 endpoints)
    xs = np.concatenate([[0.0], cum, [100.0]])
    ys = np.linspace(0.0, 100.0, len(xs))

    fig, ax = plt.subplots(figsize=figsize)
    cmap_inst = plt.get_cmap(cmap)
    colors = [cmap_inst(i) for i in range(len(plot_df))]

    # draw curve
    ax.plot(xs, ys, color='black', lw=1.25, zorder=15)

    # fill each phase area under the curve between its left and right boundaries
    left = 0.0
    for i, row in plot_df.iterrows():
        right = left + row['share_norm_pct']
        # select points on the curve between left and right (inclusive)
        mask = (xs >= left - 1e-8) & (xs <= right + 1e-8)
        xs_seg = xs[mask]
        ys_seg = ys[mask]
        # ensure segment has at least two points for fill_between
        if len(xs_seg) < 2:
            # create a tiny segment at left/right so fill_between works
            xs_seg = np.array([left, right])
            ys_seg = np.interp(xs_seg, xs, ys)
        ax.fill_between(xs_seg, ys_seg, 0, color=colors[i % len(colors)], alpha=alpha, zorder=5)

        if label_pct:
            pct = row[share_col]
            label = f"{row[phase_col]}\n{pct:.2f}%"
            x_text = left + (right - left) / 2
            ax.text(x_text, np.interp(x_text, xs, ys) + 4, label, ha='center', va='bottom', fontsize=9, zorder=20)

        left = right

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Cumulative share of time (%)')
    ax.set_ylabel('Normalized progress (%)')
    ax.set_title('Cumulative distribution of simulation phases')
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    plt.tight_layout()
    return fig, ax