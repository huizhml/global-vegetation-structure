import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from cp.constants import OKABE_ITO_PALETTE


def get_tint(color, factor=0.5):
    """Blends a given color with white to create a solid, lighter tint."""
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    # Blend the color towards white
    tint = rgb + (white - rgb) * factor
    return tint


def plot_rh_coverage_per_biome(
    calibrated_coverage: pd.DataFrame,
    initial_coverage: pd.DataFrame,
    save_path: str,
    subplot_order: list[str],
    y_axis_order: list[str],
    alpha,
    n_rows=2,
    n_cols=7,
    width=7.1,
    height=4.5,
    tint_factor=0.3,
    color=OKABE_ITO_PALETTE["blue"],
    target_line_color=OKABE_ITO_PALETTE["orange"],
):
    # ---------------------------------------------------------
    # 1. Setup the Plot (Double Column Width)
    # ---------------------------------------------------------
    # Nature double column is ~180 mm (7.1 inches).
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(width, height),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    axes = axes.flatten()
    ys_mapping = {val: idx for idx, val in enumerate(y_axis_order)}
    # ---------------------------------------------------------
    # 2. Draw the Plot
    # ---------------------------------------------------------
    for i, ax in enumerate(axes):
        if i < len(subplot_order):
            group_name = subplot_order[i]
            for y_tick in y_axis_order:
                calibrated_val = calibrated_coverage.loc[y_tick, group_name]
                initial_val = initial_coverage.loc[y_tick, group_name]
                y_pos = ys_mapping[y_tick]

                # Using your tint_factor for the initial points and line
                shaded_color = get_tint(color, factor=tint_factor)

                # Connecting line: thinner (1.0 pt)
                ax.plot(
                    [initial_val, calibrated_val],
                    [y_pos, y_pos],
                    color=shaded_color,
                    alpha=1,
                    linewidth=1.0,
                    zorder=2,
                )

                # Initial coverage: smaller marker
                ax.scatter(
                    initial_val,
                    y_pos,
                    color=shaded_color,
                    alpha=1,
                    marker="o",
                    s=15,
                    zorder=3,
                    edgecolors="none",
                )

                # Calibrated coverage: smaller diamond, very thin border
                ax.scatter(
                    calibrated_val,
                    y_pos,
                    color=color,
                    alpha=1.0,
                    marker="D",
                    s=12,
                    edgecolors="black",
                    linewidth=0.4,
                    zorder=4,
                )

            # Plot reference line at 0.9 coverage
            ax.axvline(
                0.9,
                color=target_line_color,
                linestyle="--",
                linewidth=0.5,
                alpha=1,
                zorder=1,
            )

            # ---------------------------------------------------------
            # 3. Subplot Formatting
            # ---------------------------------------------------------
            # ax.set_title(textwrap.fill(group_name, width=15), fontsize=6, pad=4)
            ax.set_title(group_name, fontsize=6, pad=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, color="#E0E0E0", linestyle="-", linewidth=0.5, zorder=0)
            ax.xaxis.grid(False)
            ax.tick_params(axis="both", labelsize=6)
        else:
            # Hide empty subplots if biomes < 14
            ax.set_visible(False)

    # ---------------------------------------------------------
    # 4. Shared Labels & Legend
    # ---------------------------------------------------------
    axes[0].set_yticks(range(len(y_axis_order)))
    axes[0].set_yticklabels(y_axis_order)

    # Shared X-label
    fig.supxlabel("Coverage", fontsize=8, y=0.05)

    # Legend (Placed at the top center of the entire figure)
    shaded_color = get_tint(color, factor=tint_factor)
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color=shaded_color.tolist(),
            marker="o",
            markeredgecolor="None",
            linestyle="None",
            markersize=3.5,
            alpha=1,
            label="Initial",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=color,
            marker="D",
            linestyle="None",
            markersize=3.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="Calibrated",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=target_line_color,
            linestyle="--",
            linewidth=0.5,
            alpha=1,
            # label=textwrap.fill("Min. target coverage", width=15),
            label=f"Min. target coverage ({100 * (1 - alpha):.0f}%)",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=6,
        handletextpad=0.4,
        columnspacing=1.5,
        labelspacing=0.8,
    )

    # ---------------------------------------------------------
    # 5. High-Res Export
    # ---------------------------------------------------------
    plt.tight_layout(w_pad=1.0, h_pad=2.0)
    plt.subplots_adjust(top=0.90, bottom=0.15)

    plt.savefig(
        save_path,
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    # plt.show()
    plt.close()
