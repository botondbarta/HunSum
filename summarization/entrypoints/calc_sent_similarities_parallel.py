import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm

from summarization.preprocess.document_embedder import DocumentEmbedder
from summarization.utils.data_helpers import is_site_in_sites, make_dir_if_not_exists
from summarization.utils.logger import get_logger

os.environ['TOKENIZERS_PARALLELISM'] = 'False'
tqdm.pandas()


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--sites', default='all', help='Sites to calculate sentence similarities for')
def main(input_dir, output_dir, sites):
    num_partitions = 3
    chunk_size = 10000
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

                partitions = [df_chunk[i::num_partitions] for i in range(num_partitions)]
                with ThreadPoolExecutor(max_workers=num_partitions) as executor:
                    processed_partitions = []
                    for i, partition in enumerate(partitions):
                        future = executor.submit(process_partition, (name, model[i], partition, logger))
                        processed_partitions.append(future)
                processed_dataframes = [future.result() for future in processed_partitions]

                # Concatenate the processed DataFrames from different partitions
                merged_dataframe = pd.concat(processed_dataframes, ignore_index=True)

                # Save the merged DataFrame to a JSON file
                merged_dataframe.to_json(f'{output_dir}/{name}/{domain}.jsonl.gz', orient='records', lines=True,
                                         compression='gzip', mode='a')

                print("All partitions have been processed and merged.")

        [model[i].to('cpu') for i in range(num_partitions)]


def process_partition(args):
    name, model, partition, logger = args

    logger.info(f'Loaded model {name}')

    partition[f'lead_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding(model, x['lead']).tolist(), axis=1)

    partition[f'lead_sent_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_sent_embedding(model, x['lead']).tolist(), axis=1)

    partition[f'article_sent_emb_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_sent_embedding(model, x['article']).tolist(), axis=1)

    partition[f'most_similar_sent_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_sent_emb_{name}'],
                                                                  x[f'article_sent_emb_{name}']).tolist(),
        axis=1)

    partition[f'most_similar_{name}'] = partition.progress_apply(
        lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_emb_{name}'],
                                                                  x[f'article_sent_emb_{name}']).tolist(),
        axis=1)

    return partition


if __name__ == '__main__':
    main()
