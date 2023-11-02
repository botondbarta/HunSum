from multiprocessing import Pool
from os import path, mkdir
from typing import List

import numpy as np
import pandas as pd


def parallelize_df_processing(df, func, num_cores=4, num_partitions=20):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df


def make_dir_if_not_exists(directory):
    if not path.exists(directory):
        mkdir(directory)


def is_site_in_sites(site: str, sites: List[str]):
    for x in sites:
        if x in site:
            return True
    return False


def get_domain_of_df_site(df):
    try:
        return df.iloc[0].domain.split('.')[0]
    except (AttributeError, TypeError):
        raise AssertionError('Input should be dataframe with at least one row')
