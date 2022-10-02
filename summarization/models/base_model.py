import os
from abc import abstractmethod, ABC

import nltk
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
            site_df = pd.read_json(file, lines=True)
            site_df = site_df[['lead', 'article']]
            site_df = self.drop_na_and_duplicates(site_df)
            site_df = site_df.astype('str')
            site_df = site_df.sample(frac=1, random_state=123)
            site_dfs.append(site_df)
        df = pd.concat(site_dfs)
        train, validate, test = np.split(df.sample(frac=1, random_state=123),
                                         [int(self.config.train_size * len(df)),
                                          int((self.config.train_size + self.config.valid_size) * len(df))])
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(train),
            'validation': Dataset.from_pandas(validate),
            'test': Dataset.from_pandas(test),
        })

        return raw_datasets

    @staticmethod
    def drop_na_and_duplicates(df):
        df = df.dropna()
        df = df.drop_duplicates(subset='article')
        return df

    def tokenize_datasets(self, raw_datasets):
        return raw_datasets.map(self.process_data_to_model_inputs, batched=True, remove_columns=['article', 'lead'])

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
