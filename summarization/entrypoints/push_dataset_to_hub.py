import glob
import os

import click
import pandas as pd
from datasets import DatasetDict, Dataset


def load_dataset(data_dir, shuffle=True):
    files = [data_dir] if os.path.isfile(data_dir) else sorted(glob.glob(f'{data_dir}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df['date_of_creation'] = pd.to_datetime(site_df['date_of_creation'])
        site_df = site_df.drop('doc_similarity', axis=1)
        site_df = site_df.drop('cc_date', axis=1)
        site_dfs.append(site_df)
    df = pd.concat(site_dfs)
    if shuffle:
        df = df.sample(frac=1, random_state=123)
    return Dataset.from_pandas(df, preserve_index=False)


def load_ext_dataset(data_dir, shuffle=True):
    files = [data_dir] if os.path.isfile(data_dir) else sorted(glob.glob(f'{data_dir}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        for chunk, df_chunk in enumerate(pd.read_json(file, lines=True, chunksize=10000)):
            site_df = df_chunk[['uuid', 'article', 'tokenized_article', 'sent-labels']]
            site_df = site_df.rename(columns={'sent-labels': 'labels'})
            site_df['tokenized_article'] = site_df['tokenized_article'].apply(
                lambda x: [s.replace('\\n\\n', ' ').replace('\\n', '').replace('" "', '').replace('""', '') for s in x])
            site_dfs.append(site_df)
    df = pd.concat(site_dfs)
    if shuffle:
        df = df.sample(frac=1, random_state=123)
    return Dataset.from_pandas(df, preserve_index=False)


@click.command()
@click.argument('data_dir')
@click.argument('hub_name')
@click.option('--ext', is_flag=True, default=False)
def main(data_dir, hub_name, ext):
    raw_datasets = DatasetDict()
    if ext:
        raw_datasets['train'] = load_ext_dataset(os.path.join(data_dir, 'train'))
        raw_datasets['validation'] = load_ext_dataset(os.path.join(data_dir, 'valid'))
        raw_datasets['test'] = load_ext_dataset(os.path.join(data_dir, 'test'))
    else:
        raw_datasets['train'] = load_dataset(os.path.join(data_dir, 'train'))
        raw_datasets['validation'] = load_dataset(os.path.join(data_dir, 'valid'))
        raw_datasets['test'] = load_dataset(os.path.join(data_dir, 'test'), shuffle=False)

    raw_datasets.push_to_hub(hub_name)


if __name__ == '__main__':
    main()
