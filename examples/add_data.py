# python script to add 'refractive_index' and
# 'abbe_number_sos' data to the database file.
import sys
import os
import numpy as np
import pandas as pd

# get num_cycle from command line
values = sys.argv
if len(values) > 1:
    num_cycle = sys.argv[1]
    pass
else:
    print("Number of cycles is not found")

# Loading of environment variables.
tmp_path = os.getenv('RadonPy_DBID')
tmp_ID = os.getenv('RadonPy_Monomer_ID')
file_path = os.path.join(tmp_path, "analyze", "results.csv")

# Load the db file.
db_path = "/path/to/db"
df_path = os.path.join(db_path, "y.csv")
df = pd.read_csv(df_path)

try:
    df_tmp = pd.read_csv(file_path)
    # Check for 'check_eq' and NaN values in 'refractive_index' and
    # 'abbe_number_sos' with a more concise conditional statement.
    if (
        df_tmp["check_eq"].iloc[0] and not (
            np.isnan(df_tmp["refractive_index"].iloc[0]) or
            np.isnan(df_tmp["abbe_number_sos"].iloc[0])
        )
    ):
        refractive_index = df_tmp["refractive_index"].iloc[0]
        abbe_number = df_tmp["abbe_number_sos"].iloc[0]
    else:
        refractive_index = abbe_number = -9999
except FileNotFoundError:
    refractive_index = abbe_number = -9999

# Update the dataframe in a more concise way by setting values
# for multiple columns at once.
df.loc[
    df['monomer_ID'] == tmp_ID,
    ['refractive_index', 'abbe_number', 'cycle']
] = [refractive_index, abbe_number, num_cycle]

# Save the updated dataframe.
df.to_csv(df_path, index=False)
