import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_utilities(
    utilities,
    confidence_intervals,
    labels,
    y_label,
    x_label,
    name,
    plot_optimal,
    optimal_utility,
    output_folder,
):
    os.makedirs(output_folder, exist_ok=True)

    utilities = [
        np.asarray(utility, dtype=float)
        for utility in utilities
    ]

    confidence_intervals = [
        np.asarray(ci, dtype=float)
        for ci in confidence_intervals
    ]

    if len(utilities) != len(labels):
        raise ValueError(
            "utilities and labels must have the same length."
        )

    if len(confidence_intervals) != len(utilities):
        raise ValueError(
            "Each utility must have a confidence interval."
        )

    lengths = [len(utility) for utility in utilities]
    ci_lengths = [len(ci) for ci in confidence_intervals]

    if len(set(lengths)) != 1:
        raise ValueError(
            "All utility sequences must have the same length."
        )

    if ci_lengths != lengths:
        raise ValueError(
            "Each confidence interval must have the same "
            "length as its utility sequence."
        )

    time = np.arange(1, lengths[0] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = {
        "Time slot": time,
    }

    for utility, ci, label in zip(
        utilities,
        confidence_intervals,
        labels,
    ):
        line, = ax.plot(
            time,
            utility,
            linewidth=2.8,
            label=label,
        )

        color = line.get_color()

        lower_bound = utility - ci
        upper_bound = utility + ci

        ax.fill_between(
            time,
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

        plot_data[label] = utility
        plot_data[f"{label} lower 95% CI"] = lower_bound
        plot_data[f"{label} upper 95% CI"] = upper_bound

    if plot_optimal:
        ax.axhline(
            optimal_utility,
            color="black",
            linestyle="--",
            linewidth=2,
            label=r"Optimal utility $\phi^*$",
        )

        plot_data["Optimal utility"] = np.full(
            len(time),
            optimal_utility,
        )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    file_name = (
        name.replace(" ", "_")
        .replace("/", "_")
    )

    pdf_path = os.path.join(
        output_folder,
        f"{file_name}.pdf",
    )

    png_path = os.path.join(
        output_folder,
        f"{file_name}.png",
    )

    csv_path = os.path.join(
        output_folder,
        f"{file_name}.csv",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    pd.DataFrame(plot_data).to_csv(
        csv_path,
        index=False,
    )

    plt.show()
    plt.close(fig)

    print("Saved:", pdf_path)
    print("Saved:", png_path)
    print("Saved:", csv_path)

    return pdf_path, png_path, csv_path

def plot_horizon_scaling(
    Hs,
    departure_utilities,
    total_queues,
    optimal_utility,
    output_folder,
    name="HorizonScaling",
):
    from matplotlib.ticker import FuncFormatter

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    Hs = np.asarray(
        Hs,
        dtype=float,
    )

    departure_utilities = np.asarray(
        departure_utilities,
        dtype=float,
    )

    total_queues = np.asarray(
        total_queues,
        dtype=float,
    )

    # Expected shape:
    # (number of horizons, number of simulations)
    if departure_utilities.shape != total_queues.shape:
        raise ValueError(
            "departure_utilities and total_queues "
            "must have the same shape."
        )

    if departure_utilities.shape[0] != len(Hs):
        raise ValueError(
            "The first dimension of the result arrays "
            "must equal len(Hs)."
        )

    num_sims = departure_utilities.shape[1]

    if num_sims < 2:
        raise ValueError(
            "At least two simulation runs are required "
            "to calculate confidence intervals."
        )

    # --------------------------------------------
    # Departure-utility shortfall
    # --------------------------------------------

    shortfall_runs = (
        optimal_utility
        - departure_utilities
    )

    shortfall_mean = np.mean(
        shortfall_runs,
        axis=1,
    )

    shortfall_std = np.std(
        shortfall_runs,
        axis=1,
        ddof=1,
    )

    shortfall_ci = (
        1.96
        * shortfall_std
        / np.sqrt(num_sims)
    )

    # --------------------------------------------
    # Terminal total queue
    # --------------------------------------------

    queue_mean = np.mean(
        total_queues,
        axis=1,
    )

    queue_std = np.std(
        total_queues,
        axis=1,
        ddof=1,
    )

    queue_ci = (
        1.96
        * queue_std
        / np.sqrt(num_sims)
    )

    # --------------------------------------------
    # Create figure
    # --------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    # ============================================
    # Departure-utility shortfall
    # ============================================

    utility_line, = axes[0].plot(
        Hs,
        shortfall_mean,
        color="#0072B2",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=2.8,
        label="Regenerative UCB",
        zorder=3,
    )

    axes[0].fill_between(
        Hs,
        shortfall_mean - shortfall_ci,
        shortfall_mean + shortfall_ci,
        color=utility_line.get_color(),
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )

    axes[0].axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        zorder=2,
    )

    # Normal linear x-axis
    axes[0].set_xscale("linear")

    axes[0].set_xlim(
        0,
        1.03 * np.max(Hs),
    )

    axes[0].xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, position: (
                f"{x / 1000:.0f}k"
                if x >= 1000
                else f"{x:.0f}"
            )
        )
    )

    axes[0].set_xlabel(
        r"Horizon $H$",
        fontsize=13,
    )

    axes[0].set_ylabel(
        r"$\phi^*-\mathbb{E}"
        r"[\phi(\bar{\mathbf{D}}(H))]$",
        fontsize=13,
    )

    axes[0].set_title(
        "Departure-utility shortfall",
        fontsize=13,
    )

    axes[0].legend(
        fontsize=11,
        framealpha=0.9,
    )

    axes[0].grid(
        alpha=0.3,
        zorder=0,
    )

    # ============================================
    # Terminal total queue
    # ============================================

    queue_line, = axes[1].plot(
        Hs,
        queue_mean,
        color="#0072B2",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=2.8,
        label="Regenerative UCB",
        zorder=3,
    )

    queue_lower = np.maximum(
        queue_mean - queue_ci,
        1e-12,
    )

    axes[1].fill_between(
        Hs,
        queue_lower,
        queue_mean + queue_ci,
        color=queue_line.get_color(),
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )

    # Log-log axes for queue scaling
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[1].set_xlabel(
        r"Horizon $H$",
        fontsize=13,
    )

    axes[1].set_ylabel(
        r"$\sum_{i=1}^{n}Q_i(H)$",
        fontsize=13,
    )

    axes[1].set_title(
        "Terminal total queue",
        fontsize=13,
    )

    axes[1].legend(
        fontsize=11,
        framealpha=0.9,
    )

    axes[1].grid(
        alpha=0.3,
        which="both",
        zorder=0,
    )

    # --------------------------------------------
    # Shared axis formatting
    # --------------------------------------------

    for ax in axes:
        ax.tick_params(
            axis="both",
            labelsize=11,
        )

    fig.tight_layout()

    # --------------------------------------------
    # Output paths
    # --------------------------------------------

    file_name = (
        name.replace(" ", "_")
        .replace("/", "_")
    )

    pdf_path = os.path.join(
        output_folder,
        f"{file_name}.pdf",
    )

    png_path = os.path.join(
        output_folder,
        f"{file_name}.png",
    )

    csv_path = os.path.join(
        output_folder,
        f"{file_name}.csv",
    )

    # --------------------------------------------
    # Save raw per-run results
    # --------------------------------------------

    csv_data = pd.DataFrame({
        "H": np.repeat(
            Hs.astype(int),
            num_sims,
        ),
        "Simulation run": np.tile(
            np.arange(1, num_sims + 1),
            len(Hs),
        ),
        "Departure utility": (
            departure_utilities.reshape(-1)
        ),
        "Departure utility shortfall": (
            shortfall_runs.reshape(-1)
        ),
        "Terminal total queue": (
            total_queues.reshape(-1)
        ),
    })

    csv_data.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------
    # Save figure
    # --------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print("Saved:", pdf_path)
    print("Saved:", png_path)
    print("Saved:", csv_path)

def plot_Kappa_sensitivity(
    Kappa_values,
    departure_utility_runs,
    total_queue_runs,
    optimal_utility,
    output_folder,
    name="Kappa_Sensitivity",
    highlight_Kappa=0.015,
):
    """
    Plot departure utility and total queue backlog for each kappa value.

    Both run arrays must have shape:
        (number of kappa values, number of runs, horizon)

    Curves show averages across runs; shaded regions show approximate
    95% confidence intervals.
    """
    Kappa_values = np.asarray(Kappa_values, dtype=float)
    departure_utility_runs = np.asarray(
        departure_utility_runs, dtype=float
    )
    total_queue_runs = np.asarray(
        total_queue_runs, dtype=float
    )

    if departure_utility_runs.ndim != 3 or total_queue_runs.ndim != 3:
        raise ValueError(
            "Both run arrays must have shape "
            "(number of kappa values, number of runs, horizon)."
        )

    if departure_utility_runs.shape != total_queue_runs.shape:
        raise ValueError(
            "departure_utility_runs and total_queue_runs "
            "must have the same shape."
        )

    num_kappa, num_sims, horizon = departure_utility_runs.shape

    if num_kappa != len(Kappa_values):
        raise ValueError(
            "The first dimension of the run arrays must equal "
            "len(Kappa_values)."
        )

    if num_sims < 2:
        raise ValueError(
            "At least two simulation runs are needed "
            "for confidence intervals."
        )

    os.makedirs(output_folder, exist_ok=True)
    time = np.arange(1, horizon + 1)

    departure_mean = np.mean(departure_utility_runs, axis=1)
    departure_ci = (
        1.96
        * np.std(departure_utility_runs, axis=1, ddof=1)
        / np.sqrt(num_sims)
    )

    queue_mean = np.mean(total_queue_runs, axis=1)
    queue_ci = (
        1.96
        * np.std(total_queue_runs, axis=1, ddof=1)
        / np.sqrt(num_sims)
    )

    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    marker_spacing = max(horizon // 12, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    departure_ax, queue_ax = axes

    csv_data = {
        "Time slot": time,
        "Optimal utility": np.full(horizon, optimal_utility),
    }

    for index, Kappa_value in enumerate(Kappa_values):
        color = colors[index % len(colors)]
        linestyle = linestyles[index % len(linestyles)]
        marker = markers[index % len(markers)]

        highlighted = (
            highlight_Kappa is not None
            and np.isclose(Kappa_value, highlight_Kappa)
        )

        linewidth = 3.2 if highlighted else 2.1
        markersize = 7.0 if highlighted else 5.5
        confidence_alpha = 0.16 if highlighted else 0.08
        zorder = 5 if highlighted else 3

        plot_label = rf"$\kappa={Kappa_value:g}$"
        csv_label = f"kappa={Kappa_value:g}"

        departure_lower = departure_mean[index] - departure_ci[index]
        departure_upper = departure_mean[index] + departure_ci[index]

        queue_lower = np.maximum(
            queue_mean[index] - queue_ci[index], 0
        )
        queue_upper = queue_mean[index] + queue_ci[index]

        departure_ax.plot(
            time,
            departure_mean[index],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_spacing,
            markersize=markersize,
            linewidth=linewidth,
            label=plot_label,
            zorder=zorder,
        )
        departure_ax.fill_between(
            time,
            departure_lower,
            departure_upper,
            color=color,
            alpha=confidence_alpha,
            linewidth=0,
            zorder=zorder - 1,
        )

        queue_ax.plot(
            time,
            queue_mean[index],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_spacing,
            markersize=markersize,
            linewidth=linewidth,
            label=plot_label,
            zorder=zorder,
        )
        queue_ax.fill_between(
            time,
            queue_lower,
            queue_upper,
            color=color,
            alpha=confidence_alpha,
            linewidth=0,
            zorder=zorder - 1,
        )

        csv_data[f"{csv_label} departure mean"] = departure_mean[index]
        csv_data[
            f"{csv_label} departure lower 95% CI"
        ] = departure_lower
        csv_data[
            f"{csv_label} departure upper 95% CI"
        ] = departure_upper

        csv_data[f"{csv_label} total queue mean"] = queue_mean[index]
        csv_data[
            f"{csv_label} total queue lower 95% CI"
        ] = queue_lower
        csv_data[
            f"{csv_label} total queue upper 95% CI"
        ] = queue_upper

    departure_ax.axhline(
        optimal_utility,
        color="black",
        linestyle=(0, (6, 3)),
        linewidth=2.3,
        label=r"Optimal utility $\phi^*$",
        zorder=6,
    )

    departure_ax.set_xlabel(r"Time slot $t$", fontsize=14)
    departure_ax.set_ylabel(
        r"$\phi(\bar{\mathbf{D}}(t))$", fontsize=14
    )
    departure_ax.set_title("Departure utility", fontsize=14)

    queue_ax.set_xlabel(r"Time slot $t$", fontsize=14)
    queue_ax.set_ylabel(
        r"$\sum_{i=1}^{n}Q_i(t)$", fontsize=14
    )
    queue_ax.set_title("Total queue backlog", fontsize=14)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25, linestyle=":")
        ax.margins(x=0)
        ax.legend(fontsize=10, ncol=2, framealpha=0.95)

    fig.tight_layout()

    file_name = name.replace(" ", "_").replace("/", "_")
    pdf_path = os.path.join(output_folder, f"{file_name}.pdf")
    png_path = os.path.join(output_folder, f"{file_name}.png")
    csv_path = os.path.join(output_folder, f"{file_name}.csv")

    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print("Saved:", pdf_path)
    print("Saved:", png_path)
    print("Saved:", csv_path)

    return pdf_path, png_path, csv_path
