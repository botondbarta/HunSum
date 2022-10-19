import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset

from summarization.utils.config_reader import get_config_from_yaml


class BaseModel(ABC):
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)

    @abstractmethod
    def process_data_to_model_inputs(self, batch):
        raise NotImplementedError

    @abstractmethod
    def full_train(self):
        raise NotImplementedError

    def load_dataset(self, data_dir):
        site_dfs = []
        for file in os.listdir(data_dir):
            site_df = pd.read_json(os.path.join(data_dir, file), lines=True)
            site_df = site_df[['lead', 'article']]
            site_df = self.drop_na_and_duplicates(site_df)
            site_df = site_df.astype('str')
            site_df = site_df.sample(frac=1, random_state=123)
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).sample(frac=1, random_state=123)

        return Dataset.from_pandas(df)

    @staticmethod
    def drop_na_and_duplicates(df):
        df = df.dropna()
        df = df.drop_duplicates(subset='article')
        return df

    def tokenize_datasets(self, raw_datasets):
        return raw_datasets.map(self.process_data_to_model_inputs, batched=True, remove_columns=['article', 'lead'])

