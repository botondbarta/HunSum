import glob
import multiprocessing as mp
from pathlib import Path

import click
import numpy as np
import pandas as pd
from torch.cuda import is_available as is_cuda_available
from transformers import AutoModel, AutoTokenizer

from summarization.utils.data_helpers import is_site_in_sites

device = 'cuda' if is_cuda_available() else 'cpu'


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--num_partitions', default=1, type=click.INT)
@click.option('--chunk_size', default=10000, type=click.INT)
@click.option('--sites', default='all', help='Help')
def main(input_dir, output_dir, num_partitions, chunk_size, sites):
    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    tokenizers = [AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc') for _ in range(num_partitions)]
    models = [AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc').to(device) for _ in range(num_partitions)]

    for site, domain in zip(sites, site_domains):
        print(f'Processing site {domain}')
        for chunk, df_chunk in enumerate(pd.read_json(site, lines=True, chunksize=chunk_size)):
            partitions = np.array_split(df_chunk, num_partitions)

            with mp.get_context('spawn').Pool(num_partitions) as pool:
                processed_partitions = pool.map(prepare_data_for_extractive, zip(models, tokenizers, partitions))

            merged_dataframe = pd.concat(processed_partitions)

            merged_dataframe.to_json(f'{output_dir}/{domain}.jsonl.gz', orient='records', lines=True,
                                     compression='gzip', mode='a')


def prepare_data_for_extractive(args):
    model, tokenizer, partition = args
    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    partition['tokenizer_input'] = partition['tokenized_article'].apply(
        lambda x: '[CLS]' + '[SEP][CLS]'.join(x) + '[SEP]')
    inputs = tokenizer(partition['tokenizer_input'].tolist(),
                       padding=True, truncation=True, max_length=512,
                       add_special_tokens=False, return_tensors="pt")

    # TODO később rákolni hogy az utolsó clst töröljük, ha több mint a sep
    cls_mask = inputs['input_ids'] == cls_token
    sep_mask = inputs['input_ids'] == sep_token
    num_of_seps = sep_mask.sum(1)

    for row, seps in zip(cls_mask, num_of_seps):
        # Find the indices of True values in the row
        true_indices = row.nonzero().squeeze()

        if len(true_indices) > seps:
            # Determine the index of the last True value in the row
            last_true_index = true_indices[-1].item()

            # Remove the last True value in the row by setting it to False
            row[last_true_index] = False

    num_of_clss = cls_mask.sum(1)
    partition['num_of_cls'] = num_of_clss.tolist()

    partition['remaining_labels'] = partition.apply(lambda x: x['labels'][:x['num_of_cls']], axis=1)
    partition['remaining_sent_labels'] = partition.apply(lambda x: x['sent-labels'][:x['num_of_cls']], axis=1)

    outputs = model(**inputs.to(device))

    cls_vectors = outputs.last_hidden_state[cls_mask]

    pd.DataFrame({'vectors': cls_vectors,
                  'labels': [item for sublist in partition['remaining_labels'].tolist() for item in sublist],
                  'sent_labels': [item for sublist in partition['remaining_sent_labels'].tolist() for item in sublist]
                  })


if __name__ == '__main__':
    main()
