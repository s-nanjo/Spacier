import pandas as pd
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.ff.descriptor import FF_descriptor


def calc_ff_descriptors(df, mp=None, n=10, nk=20):
    """
    Calculate force field descriptors for a given DataFrame of molecules.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the SMILES.
    - mp (int): Number of processors to use for parallel computation.
                Default is None.
    - n (int):  Number of monomer repetitions in cyclicpolymer.
                Default is 10.
    - nk (int): Number of kernel centers to use in the descriptor calculation.
                Default is 20.

    Returns:
    - pandas.DataFrame: DataFrame containing the calculated FF descriptors.

    """

    sigma = 1/nk/2  # Kernel width, default value
    mu = None    # Kernel center, default value

    ff_desc = FF_descriptor(GAFF2_mod(), polar=False)
    desc = ff_desc.ffkm_mp(
        df["smiles"], mp=mp, nk=nk, s=sigma, mu=mu, cyclic=n
    )
    desc_names = ff_desc.ffkm_desc_names(nk=nk)
    return pd.DataFrame(desc, columns=desc_names)
