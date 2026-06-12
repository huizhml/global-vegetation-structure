from typing import cast

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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


def nature_subplot_formatting(ax, title):
    ax.set_title(title, pad=4)  # fontsize=6
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E0E0E0", linestyle="-", linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.tick_params(axis="both", labelsize=6)


def nature_closing_figure_formatting(
    fig, legend_elements, save_path, legend_loc="outside upper center"
):
    side_locs = {"outside left", "outside right"}
    ncol = 1 if legend_loc in side_locs else 3
    fig.legend(
        handles=legend_elements,
        loc=legend_loc,
        ncol=ncol,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.5,
        labelspacing=0.8,
    )

    plt.savefig(
        save_path,
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close()


def set_supxlabel(fig, label):
    xlabel_pad_in = 0.15  # fixed gap in inches, regardless of figsize
    y_xlabel = xlabel_pad_in / fig.get_size_inches()[1]

    fig.supxlabel(label, fontsize=7, y=-y_xlabel)


def plot_rh_coverage_per_biome(
    calibrated_coverage: pd.DataFrame,
    initial_coverage: pd.DataFrame,
    save_path: str,
    subplot_order: list[str],
    y_axis_order: list[str],
    alpha,
    n_rows=2,
    n_cols=7,
    figsize=(7.1, 4.5),
    tint_factor=0.3,
    color=OKABE_ITO_PALETTE["blue"],
    target_line_color=OKABE_ITO_PALETTE["orange"],
    skip_groups=None,
    legend_loc="outside upper center",
):
    # ---------------------------------------------------------
    # 1. Setup the Plot (Double Column Width)
    # ---------------------------------------------------------
    # Nature double column is ~180 mm (7.1 inches).
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        facecolor="white",
        constrained_layout=True,
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

                # Initial coverage: smaller marker
                ax.scatter(
                    initial_val,
                    y_pos,
                    color=shaded_color,
                    alpha=1,
                    marker="o",
                    s=15,
                    zorder=4,
                    edgecolors="none",
                )
                if skip_groups is None or group_name not in skip_groups:
                    # Connecting line: thinner (1.0 pt)
                    ax.plot(
                        [initial_val, calibrated_val],
                        [y_pos, y_pos],
                        color=shaded_color,
                        alpha=1,
                        linewidth=1.0,
                        zorder=3,
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
                        zorder=5,
                    )

            # Plot reference line at 0.9 coverage
            ax.axvline(
                1 - alpha,
                color=target_line_color,
                linestyle="--",
                linewidth=1,
                zorder=2,
            )

            if skip_groups is not None and group_name in skip_groups:
                group_name = f"{group_name}\n(not calibrated)"
            nature_subplot_formatting(ax, group_name)
        else:
            # Hide empty subplots if biomes < 14
            ax.set_visible(False)

    # ---------------------------------------------------------
    # 4. Shared Labels & Legend
    # ---------------------------------------------------------
    axes[0].set_yticks(range(len(y_axis_order)))
    axes[0].set_yticklabels(y_axis_order)

    set_supxlabel(fig, "Coverage")
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}".lstrip("0"))
        )
        ax.margins(x=0.1, y=0.1)  # 15% padding on each side of the data range

    # Legend (Placed at the top center of the entire figure)
    shaded_color = get_tint(color, factor=tint_factor)
    target_cov_label = f"Min. target coverage ({100 * (1 - alpha):.0f}%)"
    if legend_loc is not None:
        target_cov_label = f"Min. target\ncoverage ({100 * (1 - alpha):.0f}%)"
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
            linewidth=1,
            alpha=1,
            # label=textwrap.fill("Min. target coverage", width=15),
            label=target_cov_label,
        ),
    ]
    xlim = axes[0].get_xlim()
    xticks = [t for t in axes[0].get_xticks() if xlim[0] <= t <= xlim[1]]
    nature_closing_figure_formatting(
        fig, legend_elements, save_path, legend_loc
    )
    return xlim, xticks


def plot_rh_width_changes_per_biome(
    initial_width: pd.DataFrame,
    calibrated_width: pd.DataFrame,
    save_path: str,
    subplot_order: list[str],
    y_axis_order: list[str],
    n_rows=2,
    n_cols=7,
    figsize=(7.1, 4.5),
    color_improved=OKABE_ITO_PALETTE["bluish_green"],
    color_worsened=OKABE_ITO_PALETTE["vermillion"],
    color_initial="#E0E0E0",
    skip_groups=None,
    legend_loc="outside upper center",
):
    # ---------------------------------------------------------
    # 1. Setup the Plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        facecolor="white",
        constrained_layout=True,
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
                init_val = cast(float, initial_width.loc[y_tick, group_name])
                cal_val = cast(float, calibrated_width.loc[y_tick, group_name])

                diff = cal_val - init_val
                y_pos = ys_mapping[y_tick]

                # Base bar (Initial width)
                ax.barh(
                    y_pos,
                    init_val,
                    color=color_initial,
                    height=0.6,
                    zorder=2,
                    edgecolor="none",
                )

                if skip_groups is None or group_name not in skip_groups:
                    max_extent = max(init_val, cal_val)
                    text_padding = init_val * 0.02

                    if diff < -1e-5:
                        ax.barh(
                            y_pos,
                            diff,
                            left=init_val,
                            color=color_improved,
                            height=0.6,
                            zorder=3,
                            edgecolor="none",
                        )
                        ax.text(
                            max_extent + text_padding,
                            y_pos,
                            "<",
                            color=color_improved,
                            va="center",
                            ha="left",
                            fontsize=5,
                            zorder=5,
                        )
                    elif diff > 1e-5:
                        ax.barh(
                            y_pos,
                            diff,
                            left=init_val,
                            color=color_worsened,
                            height=0.6,
                            zorder=3,
                            edgecolor="none",
                        )
                        ax.text(
                            max_extent + text_padding,
                            y_pos,
                            ">",
                            color=color_worsened,
                            va="center",
                            ha="left",
                            fontsize=5,
                            zorder=5,
                        )
                    else:
                        ax.text(
                            max_extent + text_padding,
                            y_pos,
                            "○",
                            color="gray",
                            va="center",
                            ha="left",
                            fontsize=5,
                            zorder=5,
                        )

            if skip_groups is not None and group_name in skip_groups:
                group_name = f"{group_name}\n(not calibrated)"
            nature_subplot_formatting(ax, group_name)
        else:
            ax.set_visible(False)

    # ---------------------------------------------------------
    # 4. Shared Labels & Legend
    # ---------------------------------------------------------
    axes[0].set_yticks(range(len(y_axis_order)))
    axes[0].set_yticklabels(y_axis_order)

    set_supxlabel(fig, "Average width (m)")

    legend_elements = [
        mpatches.Patch(facecolor=color_initial, label="Uncalibrated width"),
        mpatches.Patch(facecolor=color_improved, label="Width reduction"),
        mpatches.Patch(facecolor=color_worsened, label="Width increase"),
    ]

    nature_closing_figure_formatting(
        fig, legend_elements, save_path, legend_loc
    )


def plot_global_coverage(
    calibrated_global: pd.Series,
    initial_global: pd.Series,
    save_path: str,
    y_axis_order: list[str],
    alpha: float,
    figsize=(1.0, 4.5),
    tint_factor=0.3,
    color=OKABE_ITO_PALETTE["blue"],
    target_line_color=OKABE_ITO_PALETTE["orange"],
    xlim=(0.78, 0.96),
    xticks=(0.8, 0.9),
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ys_mapping = {val: idx for idx, val in enumerate(y_axis_order)}
    shaded_color = get_tint(color, factor=tint_factor)

    for y_tick in y_axis_order:
        calibrated_val = calibrated_global[y_tick]
        initial_val = initial_global[y_tick]
        y_pos = ys_mapping[y_tick]

        ax.scatter(
            initial_val,
            y_pos,
            color=shaded_color,
            marker="o",
            s=15,
            zorder=4,
            edgecolors="none",
        )
        ax.plot(
            [initial_val, calibrated_val],
            [y_pos, y_pos],
            color=shaded_color,
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            calibrated_val,
            y_pos,
            color=color,
            marker="D",
            s=12,
            edgecolors="black",
            linewidth=0.4,
            zorder=5,
        )

    ax.axvline(
        1 - alpha,
        color=target_line_color,
        linestyle="--",
        linewidth=1,
        zorder=2,
    )

    ax.set_yticks(range(len(y_axis_order)))
    ax.set_yticklabels(y_axis_order)
    nature_subplot_formatting(ax, "Global")
    set_supxlabel(fig, "Coverage")
    # ax.xaxis.set_major_formatter(
    #     plt.FuncFormatter(lambda x, _: f"{x:.1f}".lstrip("0"))
    # )
    # ax.set_xlim(*xlim)
    # ax.set_xticks(list(xticks))

    ax.margins(x=0.1)  # 15% padding on each side of the data range
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.2f}".lstrip("0"))
    )
    plt.savefig(
        save_path,
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close()
