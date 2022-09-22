import glob
import os.path

import click
import pandas as pd

from summarization.utils.data_helpers import parallelize_df_processing
from summarization.utils.tokenizer import Tokenizer


@click.command()
@click.argument('corpus_dir')
@click.argument('out_dir')
@click.option('--num_process', default=1, type=click.INT)
def main(corpus_dir, out_dir, num_process):
    sites = glob.glob(f'{corpus_dir}/*.jsonl.gz')
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{corpus_dir}/', '') for site in sites]

    for site, site_domain in zip(sites, site_domains):
        df_site = pd.read_json(f'{site}', lines=True)
        df_site = parallelize_df_processing(df_site, count_sentences, num_process, 100)
        df_site = parallelize_df_processing(df_site, count_tokens, num_process, 100)
        df_site = parallelize_df_processing(df_site, count_chars, num_process, 100)
        df_site = parallelize_df_processing(df_site, count_paragraph, num_process, 100)

        # remove unnecessary columns
        df_site = df_site.drop(['article', 'lead', 'cc_date'], axis=1)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_file = os.path.join(out_dir, f'{site_domain}_stat.tsv')
        df_site.to_csv(out_file, sep='\t', index=False)


def count_sentences(df):
    df['article_sent_count'] = df['article'].apply(Tokenizer.count_sentences)
    df['lead_sent_count'] = df['lead'].apply(Tokenizer.count_sentences)
    return df


def count_tokens(df):
    df['article_token_count'] = df['article'].apply(Tokenizer.count_tokens)
    df['lead_token_count'] = df['lead'].apply(Tokenizer.count_tokens)
    return df


def count_chars(df):
    df['article_char_count'] = df['article'].apply(len)
    df['lead_char_count'] = df['lead'].apply(len)
    return df


def count_paragraph(df):
    df['article_paragraph_count'] = df['article'].apply(lambda x: len([p for p in x.split('\n\n') if p]))
    return df


if __name__ == '__main__':
    main()
