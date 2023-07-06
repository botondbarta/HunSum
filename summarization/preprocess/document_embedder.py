import glob
from datetime import datetime
from os import path
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from summarization.utils.data_helpers import is_site_in_sites, get_domain_of_df_site, make_dir_if_not_exists
from summarization.utils.logger import get_logger


class DocumentEmbedder:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.out_dir = output_dir

        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def create_doc_embeddings_for_sites(self, sites):
        make_dir_if_not_exists(self.out_dir)
        log_file = path.join(self.out_dir, f'{datetime.now().strftime("%Y_%m_%d_%H:%M")}_log.txt')
        logger = get_logger('logger', log_file)

        all_jsonl_files = glob.glob(f'{self.input_dir}/*.jsonl.gz')
        sites_to_create_embedding_for = all_jsonl_files if sites == 'all' \
            else [x for x in all_jsonl_files if is_site_in_sites(Path(x).name, sites.split(','))]

        for site in sites_to_create_embedding_for:
            df_site = pd.read_json(f'{site}', lines=True)
            logger.info(f'Processing {site} {datetime.now().strftime("%Y_%m_%d_%H:%M")}')
            self.create_document_embeddings(df_site)
            logger.info(f'Processing finished at {datetime.now().strftime("%Y_%m_%d_%H:%M")}')

            df_site.to_json(f'{self.out_dir}/{get_domain_of_df_site(df_site)}.jsonl.gz', orient='records', lines=True)

    def create_document_embeddings(self, df_site):
        df_site['lead_embeddings'] = df_site.apply(
            lambda row: self.model.encode(row['lead'], convert_to_tensor=False), axis=1)
        df_site['article_embeddings'] = df_site.apply(
            lambda row: self.model.encode(row['article'], convert_to_tensor=False), axis=1)
        df_site['doc_similarity'] = df_site.apply(
            lambda row: util.cos_sim(row['lead_embeddings'], row['article_embeddings']).item(), axis=1)
