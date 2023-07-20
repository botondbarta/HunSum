import glob
from pathlib import Path

import click
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as is_cuda_available

from summarization.preprocess.document_embedder import DocumentEmbedder
from summarization.utils.data_helpers import is_site_in_sites, make_dir_if_not_exists


def calc_similarities(model, lead, article):
    cosine_scores = DocumentEmbedder.calculate_sent_similarity(model, lead, article)

    return cosine_scores.tolist()


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--sites', default='all', help='Sites to calculate sentence similarities for')
def main(input_dir, output_dir, sites):
    make_dir_if_not_exists(output_dir)

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    device = 'cuda' if is_cuda_available() else 'cpu'
    model.to(device)

    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    for site, domain in zip(sites, site_domains):
        df_site = pd.read_json(f'{site}', lines=True)
        df_site['most_similar'] = df_site.apply(lambda x: calc_similarities(model, x['lead'], x['article']), axis=1)

        df_site.to_json(f'{output_dir}/{domain}.jsonl.gz', orient='records', lines=True, compression='gzip')


if __name__ == '__main__':
    main()
