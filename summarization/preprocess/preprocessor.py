import glob
import os
from distutils.util import strtobool
from multiprocessing import cpu_count

import pandas as pd
from lsh.cache import Cache
from lsh.minhash import MinHasher

from summarization.preprocess.language_detector import LanguageDetector
from summarization.utils.config_reader import get_config_from_yaml
from summarization.utils.data_helpers import parallelize_df_processing
from summarization.utils.dateparser import DateParser
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
        site_domains = [site.replace('.jsonl.gz', '').replace(f'{self.config.src_dir}/', '')
                        for site in glob.glob('/home/bart/data/*.jsonl.gz')]

        language_detector = LanguageDetector(model_path=self.config.lang_detector_model_path)

        for site in sites:
            df_site = pd.read_json(f'{site}', lines=True)

            df_site = self._drop_non_hungarian_sentences(df_site, language_detector)
            df_site = df_site[df_site['article'].str.len() > self.config.min_article_len]
            df_site = parallelize_df_processing(df_site, self.filter_by_number_of_sentences, cpu_count() // 2, 100)

            # add fingerprint column to df
            df_site = parallelize_df_processing(df_site, self.create_fingerprints, cpu_count() // 2, 100)

            self._add_fingerprints_to_lsh(df_site)

            # saving temporary file
            self._temporarily_save_site(df_site)

        duplicates_to_drop = self._get_duplicates_to_drop(site_domains)

        # remove duplicates from the temporary files and remove them from disk
        for (domain, drops) in duplicates_to_drop.items():
            df_site = pd.read_json(f'{self.config.src_dir}/{domain}_temp.jsonl.gz', lines=True)

            df_site = df_site[~df_site.uuid.isin(drops)]
            df_site.to_json(f'{self.config.out_dir}/{domain}_dedup.jsonl.gz', orient='records',
                            lines=True, compression='gzip')

            os.remove(f'{self.config.src_dir}/{domain}_temp.jsonl.gz')

    def _temporarily_save_site(self, df):
        df_site = df.drop('fingerprint', axis=1)
        domain = self._get_domain_of_site(df_site)
        df_site.to_json(f'{self.config.src_dir}/{domain}_temp.jsonl.gz', orient='records',
                        lines=True, compression='gzip')

    def _get_domain_of_site(self, df):
        return df.iloc[0].domain.split('.')[0]

    def _drop_non_hungarian_sentences(self, df, language_detector):
        return df[df['article']
                  .apply(lambda x: x.replace('\n', ' '))
                  .map(language_detector.predict) == 'hu']

    def _get_duplicates_to_drop(self, site_domains):
        drops = {f'{domain}': [] for domain in site_domains}

        duplicates = self.lsh.get_all_duplicates(min_jaccard=self.config.min_jaccard)

        for (left, right) in duplicates:
            left_domain, left_uuid, left_date, left_has_lead = left.split('_')
            right_domain, right_uuid, right_date, right_has_lead = right.split('_')

            # drop the one with empty lead or earlier crawl time
            drop = (left_domain, left_uuid) if strtobool(left_has_lead) and not strtobool(right_has_lead) \
                else (right_domain, right_uuid) if strtobool(right_has_lead) and not strtobool(left_has_lead) \
                else (left_domain, left_uuid) if DateParser.parse(left_date) < DateParser.parse(right_date) \
                else (right_domain, right_uuid)
            drops[drop[0]].append(drop[1])

        return drops

    def _add_fingerprints_to_lsh(self, df):
        domain = self._get_domain_of_site(df)
        for i in range(len(df)):
            row = df.iloc[i]
            has_lead = True if df.iloc[i].lead != '' else False
            self.lsh.add_fingerprint(row.fingerprint, f'{domain}_{row.uuid}_{row.cc_date}_{has_lead}')

    def filter_by_number_of_sentences(self, df):
        return df[df['article'].map(Tokenizer.count_sentences) > self.config.min_article_sentences]

    def create_fingerprints(self, df):
        df['fingerprint'] = df.apply(lambda row: self.hasher.fingerprint(row['article'].encode('utf8')), axis=1)
        return df
