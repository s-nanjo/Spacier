# Optical Polymer Dataset

This directory contains datasets related to the research on optical polymers. The datasets include calibration data for refractive index and Abbe number, and results from Bayesian Optimization (BO) experiments conducted to discover new optical polymers.

## Datasets

### Calibration.csv

- **Description**: Contains calibration data used for training, correlating RadonPy computed properties with experimental values extracted from the literature.
- **Columns**:
  - `radonpy_XX`: RadonPy-calculated property values.
  - `exp_XX`: Experimental values obtained from the literature.

### BO_data.csv

- **Description**: Contains data from BO experiments to calculate the properties of optical polymers.
- **Columns**:
  - `radonpy_XX`: RadonPy-calculated property values.
  - `cycle`: The cycle number of the Bayesian Optimization process.

## Special Value Explanations

- **0 Value in `radonpy_XX`**: Indicates that the property calculation by RadonPy is pending (i.e., the data belongs to the pool at cycle number 20 where RadonPy property calculation has not been performed yet).
- **-9999 Value in `radonpy_XX`**: Signifies that the RadonPy calculation failed (either due to DFT calculation error or because the system did not reach equilibrium in the MD simulation).
