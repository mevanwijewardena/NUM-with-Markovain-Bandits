import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_main_2x3(base_folder, output_folder=None, max_points=1500):
    if output_folder is None:
        output_folder = base_folder

    algorithms = [
        "Regenerative UCB",
        "Naive slotwise UCB",
        "Known-mean Regenerative",
        "Known-transition Belief DPP",
        "Uniform",
    ]
    panels = [
        (
            "ArrivalUtility.csv",
            "Arrival utility",
            r"$\phi(\bar{\mathbf{A}}(t))$",
        ),
        (
            "DepartureUtility.csv",
            "Departure utility",
            r"$\phi(\bar{\mathbf{D}}(t))$",
        ),
        (
            "TotalQueueSize.csv",
            "Total queue backlog",
            r"$\sum_{i=1}^{n}Q_i(t)$",
        ),
    ]
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
    ]
    styles = ["-", "--", "-.", ":", "--"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 7.5), sharex="col"
    )
    legend_lines = {}
    optimal_line = None

    for row, scenario in enumerate((1, 2)):
        for col, (filename, title, ylabel) in enumerate(panels):
            csv_path = os.path.join(
                base_folder,
                f"Scenario_{scenario}",
                filename,
            )
            full_df = pd.read_csv(csv_path)

            # Downsample for drawing only. CSV values are unchanged.
            step = max(
                1,
                int(np.ceil(len(full_df) / max_points)),
            )
            indices = np.unique(
                np.r_[
                    np.arange(0, len(full_df), step),
                    len(full_df) - 1,
                ]
            )
            df = full_df.iloc[indices]
            time = df["Time slot"].to_numpy()

            ax = axes[row, col]

            for j, algorithm in enumerate(algorithms):
                line, = ax.plot(
                    time,
                    df[algorithm].to_numpy(),
                    color=colors[j],
                    linestyle=styles[j],
                    marker=markers[j],
                    markevery=max(1, len(df) // 12),
                    markersize=4.5,
                    linewidth=2.6 if j == 0 else 1.8,
                )
                ax.fill_between(
                    time,
                    df[
                        f"{algorithm} lower 95% CI"
                    ].to_numpy(),
                    df[
                        f"{algorithm} upper 95% CI"
                    ].to_numpy(),
                    color=colors[j],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )

                if algorithm not in legend_lines:
                    legend_lines[algorithm] = line

            if col < 2:
                line = ax.axhline(
                    full_df["Optimal utility"].iloc[0],
                    color="black",
                    linestyle=(0, (6, 3)),
                    linewidth=2,
                )
                if optimal_line is None:
                    optimal_line = line

            ax.set_title(f"Scenario {scenario}: {title}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, linestyle=":")
            ax.margins(x=0)
            ax.locator_params(axis="x", nbins=4)

            if row == 1:
                ax.set_xlabel(r"Time slot $t$")

    handles = [
        legend_lines[name] for name in algorithms
    ] + [optimal_line]
    labels = algorithms + [r"Optimal utility $\phi^*$"]

    # Keep legend entries in left-to-right order across two rows.
    ncol = 3
    nrows = int(np.ceil(len(handles) / ncol))
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(handles)
    ]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    # Tighter spacing gives each panel more room.
    fig.subplots_adjust(
    left=0.075, right=0.985, top=0.94, bottom=0.19,
    wspace=0.25, hspace=0.28,
    )
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(
        output_folder, "Main_Results_2x3.pdf"
    )
    png_path = os.path.join(
        output_folder, "Main_Results_2x3.png"
    )

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return pdf_path, png_path
