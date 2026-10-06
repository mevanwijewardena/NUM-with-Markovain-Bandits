import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_l_sensitivity_2x2(scenario1_csv, scenario2_csv, output_folder, max_points=1500):
    data = [pd.read_csv(scenario1_csv), pd.read_csv(scenario2_csv)]

    pattern = re.compile(r"^L=([0-9.eE+-]+) departure mean$")
    L_values = sorted({
        float(match.group(1))
        for df in data
        for column in df.columns
        if (match := pattern.fullmatch(column))
    })
    if not L_values:
        raise ValueError("No L-sensitivity columns were found in the CSV files.")

    colors = ["#0072B2", "#D55E00", "#009E73",
              "#CC79A7", "#E69F00", "#56B4E9"]
    styles = ["-", "--", "-.", ":", "-", "--"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    legend_lines = {}
    optimal_line = None

    for row, full_df in enumerate(data):
        # Sample for plotting only; the CSV values are unchanged.
        step = max(1, int(np.ceil(len(full_df) / max_points)))
        indices = np.unique(
            np.r_[np.arange(0, len(full_df), step), len(full_df) - 1]
        )
        df = full_df.iloc[indices]
        time = df["Time slot"].to_numpy()

        panels = [
            ("departure", "Departure utility",
             r"$\phi(\bar{\mathbf{D}}(t))$"),
            ("total queue", "Total queue backlog",
             r"$\sum_{i=1}^{n}Q_i(t)$"),
        ]

        for col, (metric, title, ylabel) in enumerate(panels):
            ax = axes[row, col]

            for j, L in enumerate(L_values):
                prefix = f"L={L:g}"
                mean_col = f"{prefix} {metric} mean"
                if mean_col not in df:
                    continue

                color = colors[j % len(colors)]
                line, = ax.plot(
                    time,
                    df[mean_col].to_numpy(),
                    color=color,
                    linestyle=styles[j % len(styles)],
                    marker=markers[j % len(markers)],
                    markevery=max(1, len(df) // 12),
                    linewidth=2.6 if np.isclose(L, 0.5) else 1.8,
                    markersize=5,
                )
                ax.fill_between(
                    time,
                    df[f"{prefix} {metric} lower 95% CI"].to_numpy(),
                    df[f"{prefix} {metric} upper 95% CI"].to_numpy(),
                    color=color,
                    alpha=0.12,
                )

                if metric == "departure" and L not in legend_lines:
                    legend_lines[L] = line

            if metric == "departure":
                line = ax.axhline(
                    full_df["Optimal utility"].iloc[0],
                    color="black",
                    linestyle="--",
                    linewidth=2,
                )
                if optimal_line is None:
                    optimal_line = line

            ax.set_title(f"Scenario {row + 1}: {title}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, linestyle=":")
            ax.margins(x=0)
            if row == 1:
                ax.set_xlabel(r"Time slot $t$")

    handles = [legend_lines[L] for L in L_values if L in legend_lines]
    labels = [rf"$L={L:g}$" for L in L_values if L in legend_lines]
    handles.append(optimal_line)
    labels.append(r"Optimal utility $\phi^*$")

    # Arrange the two legend rows in left-to-right numerical order.
    ncol = min(4, len(handles))
    nrows = int(np.ceil(len(handles) / ncol))
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(handles)
    ]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.94, bottom=0.20,
        wspace=0.25, hspace=0.32,
    )
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=ncol,
        frameon=False,
        fontsize=11,
    )

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, "L_Sensitivity_2x2.pdf")
    png_path = os.path.join(output_folder, "L_Sensitivity_2x2.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return pdf_path, png_path
