import glob
import os
from pathlib import Path

import click
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as is_cuda_available

from summarization.preprocess.document_embedder import DocumentEmbedder
from summarization.utils.data_helpers import is_site_in_sites, make_dir_if_not_exists

os.environ['TOKENIZERS_PARALLELISM'] = 'False'


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--sites', default='all', help='Sites to calculate sentence similarities for')
def main(input_dir, output_dir, sites):
    make_dir_if_not_exists(output_dir)

    models = {
        'minilm': SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        'labse': SentenceTransformer('sentence-transformers/LaBSE'),
    }
    device = 'cuda' if is_cuda_available() else 'cpu'

    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    for name, model in models.items():
        model.to(device)

        for site, domain in zip(sites, site_domains):
            df_site = pd.read_json(f'{site}', lines=True)

            df_site[f'lead_emb_{name}'] = df_site.apply(
                lambda x: DocumentEmbedder.calculate_embedding(model, x['lead']).tolist(), axis=1)

            df_site[f'lead_sent_emb_{name}'] = df_site.apply(
                lambda x: DocumentEmbedder.calculate_sent_embedding(model, x['lead']).tolist(), axis=1)

            df_site[f'article_sent_emb_{name}'] = df_site.apply(
                lambda x: DocumentEmbedder.calculate_sent_embedding(model, x['article']).tolist(), axis=1)

            df_site[f'most_similar_sent_{name}'] = df_site.apply(
                lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_sent_emb_{name}'],
                                                                          x[f'article_sent_emb_{name}']).tolist(),
                axis=1)

            df_site[f'most_similar_{name}'] = df_site.apply(
                lambda x: DocumentEmbedder.calculate_embedding_similarity(x[f'lead_emb_{name}'],
                                                                          x[f'article_sent_emb_{name}']).tolist(),
                axis=1)
            model.to('cpu')

            df_site.to_json(f'{output_dir}/{name}/{domain}.jsonl.gz', orient='records', lines=True, compression='gzip')


if __name__ == '__main__':
    main()
