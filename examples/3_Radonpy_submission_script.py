#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import pandas as pd
import sys
sys.path.append('../')  # Adjust path for local imports
from spacier.ml import spacier

DATA_DIR = "path/to/data"  # Replace with your data directory

# Read data from CSV files
df = pd.read_csv(f"{DATA_DIR}/target.csv")  # Load target
df_X = pd.read_csv(f"{DATA_DIR}/feature.csv")  # Load feature
df_pool = pd.read_csv(f"{DATA_DIR}/pool.csv")  # Load monomer library (pool)
df_X_pool = pd.read_csv(f"{DATA_DIR}/pool_feature.csv")  # Load pool feature

# Determine the number of cycles from the 'cycle' column,
# incremented by one for the new cycle
num_cycle = int(df["cycle"].max()) + 1
q = "cycle<" + str(num_cycle)
print(q)

# Use spacier's Bayesian Optimization (BO) for
# selecting indices based on the EHVI method
index_list = spacier.BO(
    df_X,
    df_X_pool,
    df,
    "sklearn_GP",  # The Gaussian Process model from scikit-learn
    ["refractive_index", "abbe_number"],  # Target properties for optimization
    standardization=True  # Standardize target before optimization
).EHVI(10)

# Iterate over the selected indices to prepare and submit batch jobs
for num in index_list:
    pid = df_pool.iloc[num]["monomer_ID"]  # Get the monomer ID
    smi = df_pool.iloc[num]["smiles"]  # Get the SMILES representation
    sh_name = f"{pid}.sh"  # Define shell script name using monomer ID

    # Open a new shell script file for writing
    with open(sh_name, 'w') as f:
        # Define the content of the shell script
        df_sh = [
            '#!/bin/bash\n',
            '#PBS -q batch\n',  # Job queue
            '#PBS -l nodes=1:ppn=64\n',  # CPU number request
            '#PBS -l walltime=72:00:00\n',  # Job time request
            'cd $PBS_O_WORKDIR\n',  # Change to working directory
            ' \n',
            'export PATH=$HOME/miniconda3/bin:$PATH\n',  # Set PATH for conda
            'export PYTHONPATH=$HOME/RadonPy:$PYTHONPATH\n',  # Set PYTHONPATH for RadonPy
            '. $HOME/miniconda3/etc/profile.d/conda.sh\n',  # Source conda
            'conda activate radonpy\n',  # Activate RadonPy environment
            'export FLIB_FASTOMP=FALSE\n',  # Environmental setting for FLIB
            'export FLIB_CNTL_BARRIER_ERR=FALSE\n',  # Another FLIB environmental setting
            'export LAMMPS_EXEC=lmp_mpi\n',  # LAMMPS executable setting
            ' \n',
            f'export RadonPy_DBID="{pid}"\n',  # Set database ID to PID
            f'export RadonPy_Monomer_ID="{pid}"\n',  # Confirm monomer ID
            f'export RadonPy_SMILES="{smi}"\n',  # Set SMILES for the monomer
            f'export RadonPy_Monomer_Dir={smi}/analyze\n',  # Set directory for analysis
            'export RadonPy_OMP=0\n',  # OpenMP setting
            'export RadonPy_MPI=64\n',  # MPI setting
            'export RadonPy_GPU=0\n',  # GPU setting
            'export RadonPy_OMP_Psi4=64\n',  # Psi4 OpenMP setting
            'export RadonPy_MEM_Psi4=96000\n',  # Psi4 memory setting
            'export RadonPy_RetryEQ=4\n',  # Retry setting for equilibration
            ' \n',
            'export RadonPy_Conf_MM_MPI=0\n',  # MM MPI setting
            'export RadonPy_Conf_MM_OMP=2\n',  # MM OpenMP setting
            'export RadonPy_Conf_MM_MP=4\n',  # MM multiprocessing setting
            'export RadonPy_Conf_Psi4_OMP=64\n',  # Psi4 OpenMP setting
            'export RadonPy_Conf_Psi4_MP=0\n',  # Psi4 multiprocessing setting
            ' \n',
            'python $HOME/RadonPy/sample_script/qm.py\n',  # Run QM calculation
            'python $HOME/RadonPy/sample_script/eq.py\n',  # Run equilibration
            f'python add_data.py {num_cycle}\n',  # Add data to database
            ' \n',
        ]
        # Write the shell script content
        f.writelines(df_sh)

    # Submit the job
    job_name = 'qsub ' + sh_name
    os.system(job_name)
