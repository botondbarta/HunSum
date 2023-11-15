import glob
import multiprocessing as mp
import os
from pathlib import Path

import click
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
from tqdm import tqdm

from summarization.utils.data_helpers import is_site_in_sites, make_dir_if_not_exists
from summarization.utils.logger import get_logger

tqdm.pandas()


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--num_partitions', default=1, type=click.INT)
@click.option('--chunk_size', default=10000, type=click.INT)
@click.option('--sites', default='all', help='Sites to calculate sentence similarities for')
def main(input_dir, output_dir, num_partitions, chunk_size, sites):
    make_dir_if_not_exists(output_dir)

    log_file = os.path.join(output_dir, 'log.txt')
    logger = get_logger('textrank', log_file)

    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    for site, domain in zip(sites, site_domains):
        logger.info(f'TextRanking site {domain}')

        for chunk, df_chunk in enumerate(pd.read_json(site, lines=True, chunksize=chunk_size)):
            logger.info(f'{domain} current chunk: {chunk * chunk_size}')

            if num_partitions == 1:
                processed_partition = process_partition(df_chunk)
                processed_partition.to_json(f'{output_dir}/{domain}.jsonl.gz', orient='records', lines=True,
                                            compression='gzip', mode='a')
            else:
                partitions = np.array_split(df_chunk, num_partitions)

                with mp.get_context('spawn').Pool(num_partitions) as pool:
                    processed_partitions = pool.map(process_partition, partitions)

                merged_dataframe = pd.concat(processed_partitions)

                merged_dataframe.to_json(f'{output_dir}/{domain}.jsonl.gz', orient='records', lines=True,
                                         compression='gzip', mode='a')

                print("All partitions have been processed and merged.")


def multi_hot_encode_top_k(similarity_scores, k):
    vector_length = len(similarity_scores)
    label_values = torch.Tensor(similarity_scores).topk(k).indices
    return [1 if i in label_values else 0 for i in range(vector_length)]


def pagerank(nx_graph, max_iter=100, tol=1e-6):
    try:
        return nx.pagerank(nx_graph, max_iter=max_iter, tol=tol)
    except nx.exception.PowerIterationFailedConvergence:
        return pagerank(nx_graph, max_iter * 2, tol * 1.1)


def textrank(sent_embeddings):
    sim_mat = np.zeros([len(sent_embeddings), len(sent_embeddings)])
    for i in range(len(sent_embeddings)):
        for j in range(len(sent_embeddings)):
            if i != j:
                sim_mat[i][j] = util.cos_sim(sent_embeddings[i], sent_embeddings[j])
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = pagerank(nx_graph)
    return [v for k, v in scores.items()]


def process_partition(partition):
    partition['textrank-score'] = partition.progress_apply(
        lambda x: textrank(x['article_sent_emb_minilm']), axis=1)

    partition['labels-top-3'] = partition.progress_apply(
        lambda x: multi_hot_encode_top_k(x['textrank-score'], 3), axis=1)

    partition['labels'] = partition.progress_apply(
        lambda x: multi_hot_encode_top_k(x['textrank-score'], len(x['tokenized_lead'])), axis=1)

    # keep tokenized_article, textrank-score, labels, labels-top-3
    partition = partition[['uuid', 'tokenized_article', 'textrank-score', 'labels', 'labels-top-3']]
    return partition


if __name__ == '__main__':
    main()
