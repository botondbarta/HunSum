import glob
import timeit
from multiprocessing import cpu_count

import pandas as pd
from lsh.cache import Cache
from lsh.minhash import MinHasher

from summarization.preprocess.language_detector import LanguageDetector
from summarization.utils.config_reader import get_config_from_yaml
from summarization.utils.data_helpers import parallelize_df_processing
from summarization.utils.tokenizer import Tokenizer


class Preprocessor:
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        self.hasher = MinHasher(seeds=self.config.num_of_permutations,
                                char_ngram=self.config.char_ngram,
                                hashbytes=8,
                                random_state=3)
        self.lsh = Cache(self.hasher, num_bands=self.config.num_bands)

    def preprocess(self):
        sites = glob.glob(f'{self.config.src_dir}/*.jsonl.gz')
        df_sites = [pd.read_json(f'{site}', lines=True) for site in sites]

        language_detector = LanguageDetector(model_path=self.config.lang_detector_model_path)

        # drop non-Hungarian sentences
        df_sites = [df[df['article']
                       .apply(lambda x: x.replace('\n', ' '))
                       .map(language_detector.predict) == 'hu']
                    for df in df_sites]

        df_sites = [df[df['article'].str.len() > self.config.min_article_len] for df in df_sites]
        df_sites = [df.drop_duplicates(subset=['article']) for df in df_sites]

        # filter articles by number of sentences
        df_sites = [parallelize_df_processing(df, self.filter_by_number_of_sentences, cpu_count() // 2, 100)
                    for df in df_sites]

        # add fingerprint column to dfs
        df_sites = [parallelize_df_processing(df, self.create_fingerprints, cpu_count() // 2, 100) for df in df_sites]

        self._add_fingerprints_to_lsh(df_sites)

        df_sites = self._remove_duplicates(df_sites)
        df_sites = [df.drop('fingerprint', axis=1) for df in df_sites]

        # saving the deduplicated dataframes
        for df in df_sites:
            site = df.iloc[0].domain.split('.')[0]
            df.to_json(f'{self.config.out_dir}/{site}_dedup.jsonl.gz', orient='records', lines=True, compression='gzip')

    def _remove_duplicates(self, dfs):
        drops = {f'{i}': [] for i in range(len(dfs))}

        duplicates = self.lsh.get_all_duplicates(min_jaccard=self.config.min_jaccard)

        for (left, right) in duplicates:
            (left_df, _, left_loc) = left.partition('_')
            (right_df, _, right_loc) = right.partition('_')
            left_row = dfs[int(left_df)].iloc[int(left_loc)]
            right_row = dfs[int(right_df)].iloc[int(right_loc)]

            # drop the one with empty lead or earlier crawl time
            drop = (left_df, left_row.name) if left_row.lead == '' and right_row.lead != '' \
                else (right_df, right_row.name) if right_row.lead == '' and left_row.lead != '' \
                else (left_df, left_row.name) if left_row.cc_date < right_row.cc_date \
                else (right_df, right_row.name)
            drops[drop[0]].append(int(drop[1]))

        for df_num, df_drops in drops.items():
            dfs[int(df_num)].drop(df_drops, inplace=True)
        return dfs

    def _add_fingerprints_to_lsh(self, dfs):
        for i, df in enumerate(dfs):
            for j in range(len(df)):
                self.lsh.add_fingerprint(df.iloc[j].fingerprint, f'{i}_{j}')

    def filter_by_number_of_sentences(self, df):
        return df[df['article'].map(Tokenizer.count_sentences) > self.config.min_article_sentences]

    def create_fingerprints(self, df):
        df['fingerprint'] = df.apply(lambda row: self.hasher.fingerprint(row['article'].encode('utf8')), axis=1)
        return df
