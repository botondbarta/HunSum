import glob
import os
from distutils.util import strtobool
from os import path

import numpy as np
import pandas as pd
from lsh.cache import Cache
from lsh.minhash import MinHasher
from pandarallel import pandarallel

from summarization.entrypoints.run_parse_warc_pages import make_dir_if_not_exists
from summarization.utils.config_reader import get_config_from_yaml
from summarization.utils.dateparser import DateParser
from summarization.utils.logger import get_logger


class Deduplicator:
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        pandarallel.initialize(progress_bar=True, nb_workers=self.config.num_process)
        self.hasher = MinHasher(seeds=self.config.num_of_permutations,
                                char_ngram=self.config.char_ngram,
                                hashbytes=8,
                                random_state=3)
        self.article_lsh = Cache(self.hasher, num_bands=self.config.num_bands)
        self.lead_lsh = Cache(self.hasher, num_bands=self.config.num_bands)

    def deduplicate(self):
        make_dir_if_not_exists(self.config.dedup_out_dir)
        log_file = path.join(self.config.dedup_out_dir, 'log.txt')
        logger = get_logger('preprocess', log_file)

        sites = glob.glob(f'{self.config.dedup_src_dir}/*.jsonl.gz')
        site_domains = [site.replace('.jsonl.gz', '').replace(f'{self.config.dedup_src_dir}/', '') for site in sites]

        for site in sites:
            df_site = pd.read_json(f'{site}', lines=True)

            logger.info(f'\nCreating article fingerprints for {site}, size: {len(df_site)}')
            df_site['article_fingerprint'] = df_site.parallel_apply(
                lambda row: self.hasher.fingerprint(row['article'].encode('utf8')), axis=1)

            logger.info(f'\nCreating lead fingerprints for {site}, size: {len(df_site)}')
            df_site['lead_fingerprint'] = df_site.parallel_apply(
                lambda row: self.hasher.fingerprint(row['lead'].encode('utf8')) if row['lead'] != "" else None, axis=1)

            self._add_fingerprints_to_lsh(df_site)

        duplicates_to_drop = self._get_duplicates_to_drop(site_domains)

        # remove duplicates from the domains and save them as train-valid-test
        for (domain, drops) in duplicates_to_drop.items():
            df_site = pd.read_json(f'{self.config.dedup_src_dir}/{domain}.jsonl.gz', lines=True)
            logger.info(f'Dropping {len(drops)} duplicates from {domain}')
            df_site = df_site[~df_site.uuid.isin(drops)]

            self._split_and_save_site(df_site, domain)

    def _split_and_save_site(self, df_site, domain):
        train, validate, test = np.split(df_site.sample(frac=1, random_state=123),
                                         [int(self.config.train_size * len(df_site)),
                                          int((self.config.train_size + self.config.valid_size) * len(df_site))])

        make_dir_if_not_exists(os.path.join(self.config.dedup_out_dir, 'train'))
        make_dir_if_not_exists(os.path.join(self.config.dedup_out_dir, 'valid'))
        make_dir_if_not_exists(os.path.join(self.config.dedup_out_dir, 'test'))
        train.to_json(f'{self.config.dedup_out_dir}/train/{domain}_train.jsonl.gz', orient='records',
                      lines=True, compression='gzip')
        validate.to_json(f'{self.config.dedup_out_dir}/valid/{domain}_valid.jsonl.gz', orient='records',
                         lines=True, compression='gzip')
        test.to_json(f'{self.config.dedup_out_dir}/test/{domain}_test.jsonl.gz', orient='records',
                     lines=True, compression='gzip')

    def _get_domain_of_site(self, df):
        return df.iloc[0].domain.split('.')[0]

    def _get_duplicates_to_drop(self, site_domains):
        drops = {f'{domain}': [] for domain in site_domains}
        article_duplicates = self.article_lsh.get_all_duplicates(min_jaccard=self.config.article_min_jaccard)
        lead_duplicates = self.lead_lsh.get_all_duplicates(min_jaccard=self.config.lead_min_jaccard)

        for (left, right) in article_duplicates:
            left_domain, left_uuid, left_date, left_has_lead = left.split('_')
            right_domain, right_uuid, right_date, right_has_lead = right.split('_')

            # drop the one with empty lead or earlier crawl time
            drop = (left_domain, left_uuid) if strtobool(left_has_lead) and not strtobool(right_has_lead) \
                else (right_domain, right_uuid) if strtobool(right_has_lead) and not strtobool(left_has_lead) \
                else (left_domain, left_uuid) if DateParser.parse(left_date) < DateParser.parse(right_date) \
                else (right_domain, right_uuid)
            drops[drop[0]].append(drop[1])

        for (left, right) in lead_duplicates:
            left_domain, left_uuid, left_date = left.split('_')
            right_domain, right_uuid, right_date = right.split('_')

            # drop the one with earlier crawl time
            drop = (left_domain, left_uuid) if DateParser.parse(left_date) < DateParser.parse(right_date) \
                else (right_domain, right_uuid)
            drops[drop[0]].append(drop[1])

        return drops

    def _add_fingerprints_to_lsh(self, df):
        domain = self._get_domain_of_site(df)
        for i in range(len(df)):
            row = df.iloc[i]
            has_lead = True if df.iloc[i].lead != '' else False
            self.article_lsh.add_fingerprint(row.article_fingerprint, f'{domain}_{row.uuid}_{row.cc_date}_{has_lead}')
            if row.lead_fingerprint is not None:
                self.lead_lsh.add_fingerprint(row.lead_fingerprint, f'{domain}_{row.uuid}_{row.cc_date}')
