# Spacier Library

The `spacier` library is a Python package dedicated to advancing materials science research by providing tools for efficient data exploration and analysis, particularly in the field of polymers. It leverages various sampling and optimization techniques, including Bayesian Optimization (BO), to navigate complex material spaces effectively.

## Features

- **Data Exploration:** Tools to load, visualize, and analyze materials science datasets.
- **Bayesian Optimization:** Implements BO for efficient exploration of high-dimensional material spaces.
- **Sampling Methods:** Includes Expected Improvement (EI), Probability of Improvement (PI), and Expected Hypervolume Improvement (EHVI) among others for targeted data sampling.
- **Pareto Front Exploration:** Support for multi-objective optimization to balance trade-offs between competing material properties.

## Installation

This section guides you through setting up the `spacier` library on your system, enabling you to use it for advanced data analysis in materials science.

### Prerequisites

Before you start, make sure you have installed:
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository

Clone the `spacier` repository to your local machine by executing:

```
git clone https://github.com/s-nanjo/Spacier.git
cd Spacier
```

### Step 2: Install the Package

Within the root directory of the cloned repository, where `setup.py` is located, run:

```
pip install .
```

This installs `spacier` along with its necessary dependencies. For development purposes, where changes to the code might frequently occur, use the editable mode installation:

```
pip install -e .
```

## Usage

Getting started with `spacier` is straightforward. Below is a basic example to illustrate the initial steps of utilizing the library. For comprehensive examples, including advanced functionalities and specific use cases, please refer to the [examples](./examples/) directory in this repository.

### Basic Example

To kick off your journey with `spacier`, start by importing the necessary modules. Here's a simple code snippet:

```python
from spacier.ml import spacier

# Load the data
data_path = "../spacier/data"
df_X = pd.read_csv(f"{data_path}/X.csv")
df_pool_X = pd.read_csv(f"{data_path}/X_pool.csv")

df = pd.read_csv(f"{data_path}/y.csv")
df_pool = pd.read_csv(f"{data_path}/y_pool.csv")

# Utilize Probability of Improvement (PI)
new_index = spacier.BO(df_X, df_pool_X , df, df_pool, "sklearn_GP", ["Cp"]).PI([[3000, 4000]], 10)
print(new_index)
```

## License

Spacier is licensed under the BSD 3-Clause License. See the `LICENSE` file for more details.

## Contact

For questions or support, please contact nanjos@ism.ac.jp.
