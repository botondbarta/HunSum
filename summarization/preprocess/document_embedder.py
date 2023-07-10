import glob
from datetime import datetime
from os import path
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util

from summarization.utils.config_reader import get_config_from_yaml
from summarization.utils.data_helpers import is_site_in_sites, get_domain_of_df_site, make_dir_if_not_exists
from summarization.utils.logger import get_logger


class DocumentEmbedder:
    def __init__(self, config_path):
        self.config = get_config_from_yaml(config_path)
        self.input_dir = self.config.calc_sim_src_dir
        self.out_dir = self.config.calc_sim_out_dir

        self.model = SentenceTransformer(self.config.sim_model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def calculate_doc_similarity_for_sites(self, sites: str):
        make_dir_if_not_exists(self.out_dir)
        log_file = path.join(self.out_dir, f'{datetime.now().strftime("%Y_%m_%d_%H:%M")}_log.txt')
        logger = get_logger('logger', log_file)

        all_jsonl_files = glob.glob(f'{self.input_dir}/*.jsonl.gz')
        sites_to_create_embedding_for = all_jsonl_files if sites == 'all' \
            else [x for x in all_jsonl_files if is_site_in_sites(Path(x).name, sites.split(','))]

        for site in sites_to_create_embedding_for:
            df_site = pd.read_json(site, lines=True)
            logger.info(f'Processing {site} {datetime.now().strftime("%Y_%m_%d %H:%M")}')
            self.calculate_doc_similarities(df_site)
            logger.info(f'Processing finished at {datetime.now().strftime("%Y_%m_%d %H:%M")}')

            df_site.to_json(f'{self.out_dir}/{get_domain_of_df_site(df_site)}.jsonl.gz', orient='records', lines=True)

    def calculate_doc_similarities(self, df_site: DataFrame):
        df_site['doc_similarity'] = df_site.apply(
            lambda row: self.calculate_text_similarity(row['lead'], row['article']), axis=1)

    def calculate_text_similarity(self, lead: str, article: str):
        return util.cos_sim(
            self.model.encode(lead),
            self.model.encode(article)).item()
