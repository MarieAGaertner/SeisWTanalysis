#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot HDBSCAN cluster distribution for different hyperparameter combinations.

This script loads HDBSCAN label arrays (saved as npz files) and visualizes
cluster distributions for all prior calculated combinations of:
    - min_samples
    - min_cluster_size

Author: Marie A. Gärtner
Date: 23.10.2023 
"""

import os
import logging
import configparser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from calc_hdbscan import fig_setup, setup_logging



# =============================================================================
# Utilities
# =============================================================================
def load_config(config_path: Path) -> configparser.ConfigParser:
    """Load INI configuration safely."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def load_hdbscan_labels(filepath: Path) -> np.ndarray:
    """Load HDBSCAN labels from an .npz file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing HDBSCAN file: {filepath}")

    with np.load(filepath, allow_pickle=True) as f:
        return f["hdbscan_labels"]



# =============================================================================
# Main plot routine
# =============================================================================
def plot_cluster_distributions(
    min_samples: list[int],
    min_cluster_sizes: list[int],
    hdbscan_dir: Path,
    fig_dir: Path,
    fig_format: str = "pdf",
    fig_width: int = 426,
    fontsize: int = 8,
    dpi: int = 300,
):
    """
    Create log-scaled histograms for each (min_samples, min_cluster_size) pair.
    """

    fig = fig_setup(width=fig_width, height=0.75 * fig_width)
    axs = fig.subplots(
        len(min_cluster_sizes),
        len(min_samples),
        sharex=True,
        sharey=True,
    )

    max_n_clusters = 0
    cluster_inhabitance = []
    min_cluster_stats = []

    # -------------------------------------------------------------------------
    # Loop over hyperparameters
    # -------------------------------------------------------------------------
    for i, mcs in enumerate(min_cluster_sizes):
        for j, ms in enumerate(min_samples):

            hstr = f"minC_{mcs}_minS_{ms}" 
            hdbscan_path = hdbscan_dir / f"hdbscan_{hstr}.npz"

            logging.info(f"Loading {hdbscan_path.name}")
            labels = load_hdbscan_labels(hdbscan_path)

            max_n_clusters = max(max_n_clusters, labels.max())

            # Histogram
            x_ticks = np.arange(-1, labels.max() + 2)
            N, bins, patches = axs[i, j].hist(
                x=labels,
                bins=x_ticks,
                edgecolor="white",
                color="darkslategray",
                align="left",
            )

            axs[i, j].grid()
            axs[i, j].set_yscale("log")
            axs[i, j].set_yticks([1e1, 1e3, 1e5])
            axs[i, j].tick_params(axis="both", labelsize=fontsize)

            # Determine smallest cluster
            cluster_inhabitance.append(N)
            min_cluster_stats.append([np.min(N), ms, mcs])

            # Titles / axis labels
            if i == 0:
                axs[i, j].set_title(f"ms = {ms}", fontsize=fontsize)
            if j == 0:
                axs[i, j].set_ylabel(f"mcs = {mcs}\nCounts", fontsize=fontsize)

    # -------------------------------------------------------------------------
    # Add annotations (percentiles, smallest cluster)
    # -------------------------------------------------------------------------
    idx = 0
    xlim = axs[0, 0].get_xlim()
    ylim = axs[0, 0].get_ylim()

    for i, mcs in enumerate(min_cluster_sizes):
        for j, ms in enumerate(min_samples):
            N = cluster_inhabitance[idx]
            min_count = min_cluster_stats[idx][0]

            # Median line
            p50 = np.percentile(N, 50)

            axs[i, j].hlines(
                y=p50,
                xmin=xlim[0],
                xmax=xlim[1],
                lw=0.5,
                ls="-",
                colors="firebrick",
            )

            axs[i, j].text(
                xlim[1],
                ylim[1],
                f"min: {min_count}",
                color="darkslategray",
                va="top",
                ha="right",
                backgroundcolor="white",
                fontsize=fontsize - 4,
            )

            # X-tick formatting
            if max_n_clusters > 10:
                axs[i, j].set_xticks(np.arange(-1, max_n_clusters + 1, 3))
                axs[i, j].set_xticklabels(
                    np.arange(-1, max_n_clusters + 1, 3), rotation="vertical"
                )
            else:
                axs[i, j].set_xticks(np.arange(-1, max_n_clusters + 1))

            idx += 1

    fig.align_ylabels()

    # Save output
    fig_dir.mkdir(parents=True, exist_ok=True)
    savepath = fig_dir / f"hdbscan_param_test_hist.{fig_format}" 
    fig.savefig(savepath, format=fig_format, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    logging.info(f"Figure saved: {savepath}")


# =============================================================================
# Setup
# =============================================================================

def main():
    setup_logging()
    plt.rcParams["date.converter"] = "concise"

    # -------------------------------------------------------------------------
    # Paths & config
    # -------------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    config_path = root / "config.ini"
    cfg = load_config(config_path)

    station_dir = root.parent / cfg["seismogram"]["station"] 
    fig_dir = station_dir / "Figures"

    # -------------------------------------------------------------------------
    # HDBSCAN hyperparameters
    # -------------------------------------------------------------------------
    min_samples = [50, 50]#[10, 25, 50, 100, 200, 400]
    min_cluster_sizes = [100, 100]#200, 400, 700, 800, 900, 1000]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    plot_cluster_distributions(
        min_samples=min_samples,
        min_cluster_sizes=min_cluster_sizes,
        hdbscan_dir=station_dir,
        fig_dir=fig_dir,
    )

    logging.info("END")


if __name__ == "__main__":
    main()
# ------------------------------- END ----------------------------- #

