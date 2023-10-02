import glob
import multiprocessing as mp
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm

from summarization.preprocess.document_embedder import DocumentEmbedder
from summarization.utils.data_helpers import is_site_in_sites, make_dir_if_not_exists
from summarization.utils.logger import get_logger
from summarization.utils.tokenizer import Tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'False'
tqdm.pandas()


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--num_partitions', default=1, type=click.INT)
@click.option('--chunk_size', default=10000, type=click.INT)
@click.option('--sites', default='all', help='Sites to calculate sentence similarities for')
def main(input_dir, output_dir, num_partitions, chunk_size, sites):
    models = {
        'minilm': [SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                   for i in range(num_partitions)],
        'labse': [SentenceTransformer('sentence-transformers/LaBSE') for i in range(num_partitions)],
    }
    device = 'cuda' if is_cuda_available() else 'cpu'

    make_dir_if_not_exists(output_dir)
    for name, model in models.items():
        make_dir_if_not_exists(os.path.join(output_dir, name))

    log_file = os.path.join(output_dir, 'log.txt')
    logger = get_logger('preprocess', log_file)

    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    for name, model in models.items():
        [model[i].to(device) for i in range(num_partitions)]
        logger.info(f'Loaded model {name}')

        for site, domain in zip(sites, site_domains):
            logger.info(f'Processing site {domain}')
            for chunk, df_chunk in enumerate(pd.read_json(site, lines=True, chunksize=chunk_size)):
                logger.info(f'{domain} current chunk: {chunk * chunk_size}')

                partitions = np.array_split(df_chunk, num_partitions)

                arg_list = [(name, mod, partition, logger) for mod, partition in zip(model, partitions)]
                with mp.get_context('spawn').Pool(num_partitions) as pool:
                    processed_partitions = pool.map(process_partition, arg_list)

                # Concatenate the processed DataFrames from different partitions
                merged_dataframe = pd.concat(processed_partitions)

                # Save the merged DataFrame to a JSON file
                merged_dataframe.to_json(f'{output_dir}/{name}/{domain}.jsonl.gz', orient='records', lines=True,
                                         compression='gzip', mode='a')

                print("All partitions have been processed and merged.")

        [model[i].to('cpu') for i in range(num_partitions)]


def multi_hot_encode(similarity_scores):
    vector_length = len(similarity_scores[0])
    label_values = linear_sum_assignment(similarity_scores, maximize=True)[1]
    return [1 if i in label_values else 0 for i in range(vector_length)]


def multi_hot_encode_top_k(similarity_scores, k):
    vector_length = len(similarity_scores[0])
    label_values = torch.Tensor(similarity_scores).topk(k).indices[0]
    return [1 if i in label_values else 0 for i in range(vector_length)]


def process_partition(args):
    name, model, partition, logger = args

    partition['tokenized_lead'] = partition['lead'].progress_apply(Tokenizer.sentence_tokenize)
    partition['tokenized_article'] = partition['article'].progress_apply(Tokenizer.sentence_tokenize)

    partition[f'lead_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding(model, x['lead']).tolist(), axis=1)

    partition[f'article_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding(model, x['article']).tolist(), axis=1)

    partition[f'lead_sent_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding(model, x['tokenized_lead']).tolist(), axis=1)

    partition[f'article_sent_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding(model, x['tokenized_article']).tolist(), axis=1)

    partition[f'most_similar_sent_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_sent_emb_{name}'],
                                                                  x[f'article_sent_emb_{name}']).tolist(),
        axis=1)

    partition[f'most_similar_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_emb_{name}'],
                                                                  x[f'article_sent_emb_{name}']).tolist(),
        axis=1)

    partition['labels'] = partition.progress_apply(
        lambda x: multi_hot_encode_top_k(x[f'most_similar_{name}'], len(x['tokenized_lead'])), axis=1)
    partition['labels-top-3'] = partition.apply(
        lambda x: multi_hot_encode_top_k(x[f'most_similar_{name}'], 3), axis=1)
    partition['sent-labels'] = partition[f'most_similar_sent_{name}'].progress_apply(multi_hot_encode)

    return partition


if __name__ == '__main__':
    main()
