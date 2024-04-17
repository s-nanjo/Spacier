import numpy as np
import pandas as pd
from radonpy.core import poly
from smipoly.smip import monc, polg


def generate_pool(df, props=[]):
    """
    Generates a pool of unique polymers from a dataframe of SMILES strings.

    Parameters:
        df (pd.DataFrame): DataFrame containing molecular data.
        props (list): Additional properties to calculate for each polymer.

    Returns:
        pd.DataFrame: DataFrame with unique polymers and their properties.
    """
    _ = monc.moncls(df=df, smiColn='smiles', dsp_rsl=False)
    _ = polg.biplym(df=_, Pmode='a', dsp_rsl=False)
    _ = _[_['polym'].apply(lambda x: x.count('*') == 2)]
    df_ = _.drop_duplicates(subset=['polym']).reset_index(drop=True)
    _ = poly.full_match_smiles_listself(pd.Series(df_["polym"]), mp=2)
    _ = pd.DataFrame(_, columns=["idx1", "idx2"])

    del_idx = _["idx1"].values.tolist()
    not_del_idx = list(df_.drop(del_idx).index)
    df_ = df_.iloc[not_del_idx]
    df_ = df_.sample(frac=1).reset_index(drop=True)

    df_["monomer_ID"] = ["SMiPoly_VL" + str(_) for _ in np.arange(len(df_))]
    df_["smiles"] = df_["polym"]
    df_["cycle"] = ""

    for prop in props:
        df_[prop] = 0

    return df_[["monomer_ID", "smiles"] + props + ["cycle"]]
