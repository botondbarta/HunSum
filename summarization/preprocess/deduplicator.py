import glob
import multiprocessing as mp
from distutils.util import strtobool
from os import path

import numpy as np
import pandas as pd
from lsh.cache import Cache
from lsh.minhash import MinHasher
from tqdm import tqdm

from summarization.utils.config_reader import get_config_from_yaml
from summarization.utils.data_helpers import get_domain_of_df_site
from summarization.utils.data_helpers import make_dir_if_not_exists
from summarization.utils.dateparser import DateParser
from summarization.utils.logger import get_logger

tqdm.pandas()


class Deduplicator:
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        self.num_process = self.config.num_process
        self.hasher = MinHasher(seeds=self.config.num_of_permutations,
                                char_ngram=self.config.char_ngram,
                                hashbytes=8,
                                random_state=3)
        self.article_lsh = Cache(self.hasher, num_bands=self.config.article_num_bands)
        self.lead_lsh = Cache(self.hasher, num_bands=self.config.lead_num_bands)

    def deduplicate(self):
        make_dir_if_not_exists(self.config.dedup_out_dir)
        make_dir_if_not_exists(self.config.fingerprint_dir)

        log_file = path.join(self.config.dedup_out_dir, 'log.txt')
        logger = get_logger('preprocess', log_file)

        sites = glob.glob(f'{self.config.dedup_src_dir}/*.jsonl.gz')
        site_domains = [site.replace('.jsonl.gz', '').replace(f'{self.config.dedup_src_dir}/', '') for site in sites]

        for (site, domain) in zip(sites, site_domains):
            fingerprint_path = f'{self.config.fingerprint_dir}/{domain}.jsonl.gz'

            # check if site not in fingerprint_dir
            if not path.exists(fingerprint_path):
                logger.info(f'Creating fingerprints for {site}')
                df_site = self._create_and_save_fingerprints_for_site(site, domain)
            else:
                logger.info(f'Loading fingerprints for {site}')
                df_site = pd.read_json(fingerprint_path, lines=True)
                assert 'article_fingerprint' in df_site.columns and 'lead_fingerprint' in df_site.columns

            self._add_fingerprints_to_lsh(df_site)

        duplicates_to_drop = self._get_duplicates_to_drop(site_domains)

        # remove duplicates from the domains and save them as train-valid-test
        for (domain, drops) in duplicates_to_drop.items():
            df_site = pd.read_json(f'{self.config.dedup_src_dir}/{domain}.jsonl.gz', lines=True)
            logger.info(f'Dropping {len(drops)} duplicates from {domain}')
            df_site = df_site[~df_site.uuid.isin(drops)]

            df_site.to_json(f'{self.config.dedup_out_dir}/{domain}.jsonl.gz', orient='records',
                            lines=True, compression='gzip')

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

        for site in drops:
            drops[site] = list(set(drops[site]))
        return drops

    def _add_fingerprints_to_lsh(self, df):
        domain = get_domain_of_df_site(df)
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            has_lead = True if df.iloc[i].lead != '' else False
            self.article_lsh.add_fingerprint(row.article_fingerprint, f'{domain}_{row.uuid}_{row.cc_date}_{has_lead}')
            if row.lead_fingerprint is not None:
                self.lead_lsh.add_fingerprint(row.lead_fingerprint, f'{domain}_{row.uuid}_{row.cc_date}')

    def _create_and_save_fingerprints_for_site(self, site, domain):
        df_site = pd.read_json(site, lines=True)

        partitions = np.array_split(df_site, self.num_process)
        with mp.get_context('spawn').Pool(self.num_process) as pool:
            processed_partitions = pool.map(self._create_fingerprints, partitions)

        merged_dataframe = pd.concat(processed_partitions)
        merged_dataframe.to_json(f'{self.config.fingerprint_dir}/{domain}.jsonl.gz', orient='records', lines=True,
                                 compression='gzip')

        return merged_dataframe

    def _create_fingerprints(self, df):
        hasher = MinHasher(seeds=self.config.num_of_permutations,
                           char_ngram=self.config.char_ngram,
                           hashbytes=8,
                           random_state=3)
        df['article_fingerprint'] = df.progress_apply(
            lambda row: hasher.fingerprint(row['article'].encode('utf8')), axis=1)
        df['lead_fingerprint'] = df.progress_apply(
            lambda row: hasher.fingerprint(row['lead'].encode('utf8')) if row['lead'] != "" else None, axis=1)
        return df
