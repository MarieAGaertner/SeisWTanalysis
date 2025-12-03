## General remarks 
We used these Python scripts to analyse seismic wind turbine emissions for our publication:

Marie A. Gärtner, René Steinmann, Laura Gassner, Joachim R. R. Ritter (in review): Machine learning for data-driven pattern recognition of seismic wind turbine emissions.

These scripts were developed specifically for the analysis presented in the publication and are not actively maintained.


## Workflow
All used parameters are in the configuration file "config.ini".

### scattransform.py
The class "scattransform"  performs the loading, preprocessing, and scattering transformation. It saves an xarray including the zeros-, first-, and second-order scattering coefficients.
The first-order scattering coefficients are the time-averaged wavelet features discussed in the publication. Run this script via "main.py".

For questions regarding the scattering transform, we refer to https://scatseisnet.readthedocs.io/en/latest/#

### calc_hdbscan.py
This script performs the clustering of the time-averaged wavelet features using HDBSCAN and plots the condensed tree (compare Fig. 6). The hyperparameters for clustering can be entered in the configuration file "config.ini".

### plot_hdbscan_param_test.py
This script plots the histograms used for the HDBSCAN hyperparameter tuning (Appendix C, Fig. A3). Each hyperparameter needs to be computed with the "calc_hdbscan" script 

## General remarks

We used these Python scripts to analyse seismic wind turbine emissions
for our publication:

Marie A. Gärtner, René Steinmann, Laura Gassner, Joachim R. R. Ritter (in review): Machine learning for data-driven pattern recognition of seismic wind turbine emissions.

## Workflow
All parameters used in the workflow are defined in the configuration file: `config.ini`.

The full workflow consists of three main scripts:
### 1. `scattransform.py`

The class **scattransform** performs:
- Loading of seismic waveform data
- Preprocessing of the waveform data
- Computation of the scattering transform
- Saving an 'xarray.Dataset' containing
  - zeroth-order scattering coefficients
  - first-order scattering coefficients
  - second-order scattering coefficients.
    
The first-order scattering coefficients are the time-averaged wavelet
features discussed in the publication.\
Run this script through `main.py`.

For details about the scattering transform implementation, we refer to the official documentation:\
https://scatseisnet.readthedocs.io/en/latest/#

### 2. `calc_hdbscan.py`

This script performs unsupervised clustering of the time-averaged wavelet
features using the HDBSCAN algorithm.
It:
- Loads the precomputed time-averaged wavelet features
- Applies HDBSCAN with user-defined hyperparameters
- Saves the resulting cluster labels
- Produces the condensed tree plot (compare Fig. 6 in the publication)

The HDBSCAN hyperparameters can be adjusted in `config.ini`.

For questions regarding hdbscan, we refer to:\
https://hdbscan.readthedocs.io/en/latest/

### 3. `plot_hdbscan_param_test.py`

This script plots the histograms used for the HDBSCAN hyperparameter
tuning shown in Appendix C, Fig. A3.\

Each hyperparameter combination must be computed beforehand with the 'calc_hdbscan.py' script.
