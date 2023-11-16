import os
from pathlib import Path

import huspacy
import numpy as np
import pandas as pd
import click
from tqdm import tqdm
import multiprocessing as mp

tqdm.pandas()

def get_files(folder):
    files = []
    for file in os.listdir(folder):
        files.append(os.path.join(folder, file))
    return files


@click.command()
@click.argument('data_folder')
@click.argument('out_folder')
@click.option('--num_partitions', default=1, type=click.INT)
def main(data_folder, out_folder, num_partitions):
    files = get_files(data_folder)

    for file in files:
        df = pd.read_json(file, lines=True)
        domain = Path(file).name.replace('.jsonl.gz', '')
        partitions = np.array_split(df, num_partitions)

        with mp.get_context('spawn').Pool(num_partitions) as pool:
            processed_partitions = pool.map(process_partition, partitions)

        merged_dataframe = pd.concat(processed_partitions)
        merged_dataframe.to_json(f'{out_folder}/{domain}.jsonl.gz', orient='records', lines=True,
                                 compression='gzip', mode='a')

        df.to_json(os.path.join(out_folder, Path(file).name), orient='records', lines=True, compression='gzip')


def process_partition(partition):
    nlp = huspacy.load('hu_core_news_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", ])
    partition['entities'] = partition['article'].progress_apply(lambda x: [ent.lemma_ for ent in nlp(x).ents])
    return partition


if __name__ == '__main__':
    main()
