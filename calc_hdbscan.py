#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDBSCAN clustering workflow for wavelet-based features.

This module loads wavelet-based features from an xarray Dataset,
performs clustering using HDBSCAN, and optionally plots the
condensed cluster tree.

All configuration (paths, station ID, parameters) is read
from a config.ini file.

Author: Marie Gärtner
Date: 12.02.2024
"""
import logging
import configparser
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import hdbscan
import cmcrameri

# =============================================================================
# Utiles
# =============================================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_wavelet_features(path: Path) -> np.ndarray:
    """
    Load log10-transformed wavelet features from an xarray Dataset.
    NaN rows are removed automatically.

    Parameters
    ----------
    path : Path
        Path to the NetCDF file containing wavelet features.

    Returns
    -------
    np.ndarray
        Clean 2D array of wavelet features.
    """
    logging.info(f"Loading wavelet feature dataset: {path}")
    dset = xr.open_dataset(path)

    wavelet_features = np.log10(dset["S1"].data)
    wavelet_features = wavelet_features.reshape(wavelet_features.shape[0], -1)

    mask_valid = ~np.isnan(np.sum(wavelet_features, axis=1))

    return wavelet_features[mask_valid, :], dset['time'].values[mask_valid]


def run_hdbscan(
    data: np.ndarray,
    min_samples: int,
    min_cluster_size: int,
) -> hdbscan.HDBSCAN:
    """
    Perform HDBSCAN clustering.

    Parameters
    ----------
    data : np.ndarray
        Input feature matrix.
    min_samples : int
    min_cluster_size : int

    Returns
    -------
    hdbscan.HDBSCAN
        Fitted clusterer object.
    """
    logging.info("Performing HDBSCAN clustering")
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit(data)

    logging.info("HDBSCAN clustering finished")
    logging.info(f"Number of clusters: {len(np.unique(clusterer.labels_))}")
    logging.info(f"Cluster labels: {np.unique(clusterer.labels_)}")

    return clusterer


def save_labels(path: Path, labels: np.ndarray, times=None) -> None:
    """
    Save labels to NPZ file.

    Parameters
    ----------
    path : Path
        Output NPZ file path.
    labels : np.ndarray
    times : np.ndarray or None
        Optional timestamps.
    """
    logging.info(f"Saving labels: {path}")
    np.savez(path, hdbscan_labels=labels, times=times)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def fig_setup(
    width: float | None = None,
    height: float | None = None,
    unit: str = "pt",
    dpi: int = 300,
    layout: str = "constrained",
):
    # Conversion factors for different units to inches
    unit_to_inches = {'pt': 1 / 72, 'cm': 1 / 2.54, 'in': 1}

    # Check if the unit is supported
    if unit not in unit_to_inches:
        raise AttributeError(f"'{unit}' is not implemented.")

    # Convert width and height to inches
    width_in_inches = width * unit_to_inches[unit] if width else None
    height_in_inches = height * unit_to_inches[unit] if height else None

    fig = plt.figure()
    fig.set_layout_engine(layout)
    if width:
        fig.set_figwidth(width_in_inches)
    if height:
        fig.set_figheight(height_in_inches)
    fig.set_dpi(dpi)

    return fig

def plot_condensed_tree(
    clusterer: hdbscan.HDBSCAN,
    fig_path: Path,
    width_pt: float = 426,
    fontsize: int = 8,
    dpi: int = 300,
) -> None:
    """
    Plot a condensed tree of the HDBSCAN clustering.

    Parameters
    ----------
    clusterer : hdbscan.HDBSCAN
    outpath : Path
        Path where figure will be saved.
    width_pt : float, default 426
        Figure width in points.
    fontsize : int, default 8
        Font size for axes labels.
    dpi : int, default 300
    """
    labels_unique = np.unique(clusterer.labels_)
    logging.info("Plotting condensed HDBSCAN tree")

    # Prepare colormaps
    if labels_unique[0] == -1:
        n_cluster = labels_unique.shape[0] - 1
    else:
        n_cluster = labels_unique.shape[0]

    # Colormap for selection
    cmap_tree = plt.get_cmap("cmc.hawaii", n_cluster + 1)(np.linspace(0, 1, n_cluster + 1))[:-1]

    # Colormap for dendrogram
    colors_dendro = plt.get_cmap("cmc.grayC", 300).colors
    cmap_dendrogram = colors.ListedColormap(colors_dendro[:250, ...])

    fig = fig_setup(width=0.55 * width_pt, height=0.4 * width_pt)
    ax = fig.gca()

    clusterer.condensed_tree_.plot(
        axis=ax,
        select_clusters=True,
        log_size=True,
        selection_palette=cmap_tree,
        cmap=cmap_dendrogram,
    )
    ax.tick_params(axis="both", labelsize=fontsize)

    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved condensed tree: {fig_path}")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run_hdbscan_workflow(config_path: str | Path) -> None:
    """
    Complete HDBSCAN workflow:
    - read config
    - load features
    - cluster
    - save labels
    - plot condensed tree

    Parameters
    ----------
    config_path : str or Path
        Path to config.ini
    """

    config = configparser.ConfigParser()
    config.read(config_path)

    root = Path(config_path).parent
    station = config["seismogram"]["station"]

    # Construct paths 
    wavelet_feature_path = root.parent / station / config["directories"]["sc_str"]
    hdbscan_dir = wavelet_feature_path.parent

    fig_dir = hdbscan_dir / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    min_cluster_size = config["hdbscan"].getint("min_cluster_size")
    min_samples = config["hdbscan"].getint("min_samples")

    # Build output label filename
    hstr = f"minC_{min_cluster_size}_minS_{min_samples}" 
    labels_path = hdbscan_dir / f"hdbscan_{hstr}.npz"

    # Load wavelet features
    wavelet_features, times = load_wavelet_features(wavelet_feature_path)

    # Cluster
    clusterer = run_hdbscan(
        wavelet_features,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )

    # Save labels
    save_labels(labels_path, clusterer.labels_, times)

    # Plot tree
    fig_path = fig_dir / f"hdbscan_dendrogram_{hstr}.pdf"
    plot_condensed_tree(clusterer, fig_path)

    logging.info("Workflow completed.")

def main():
    setup_logging()
    root = Path(__file__).resolve().parent
    config_path = root / "config.ini"

    run_hdbscan_workflow(config_path)


if __name__ == "__main__":
    main()
