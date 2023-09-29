import glob
from pathlib import Path

import click
import pandas as pd
from torch.cuda import is_available as is_cuda_available
from transformers import AutoModel, AutoTokenizer

from summarization.utils.data_helpers import is_site_in_sites

device = 'cuda' if is_cuda_available() else 'cpu'


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--chunk_size', default=10000, type=click.INT)
@click.option('--sites', default='all', help='Help')
def main(input_dir, output_dir, chunk_size, sites):
    all_sites = glob.glob(f'{input_dir}/*.jsonl.gz')
    sites = all_sites if sites == 'all' else [x for x in all_sites if is_site_in_sites(Path(x).name, sites.split(','))]
    site_domains = [site.replace('.jsonl.gz', '').replace(f'{input_dir}/', '') for site in sites]

    tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
    model = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')

    for site, domain in zip(sites, site_domains):
        print(f'Processing site {domain}')
        for chunk, df_chunk in enumerate(pd.read_json(site, lines=True, chunksize=chunk_size)):
            processed_partition = prepare_data_for_extractive(df_chunk, model, tokenizer)
            processed_partition.to_json(f'{output_dir}/{domain}.jsonl.gz', orient='records', lines=True,
                                        compression='gzip', mode='a')


def prepare_data_for_extractive(df_chunk, model, tokenizer):
    model.to(device)
    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    df_chunk['tokenizer_input'] = df_chunk['tokenized_article'].apply(
        lambda x: '[CLS]' + '[SEP][CLS]'.join(x) + '[SEP]')
    inputs = tokenizer(df_chunk['tokenizer_input'].tolist(),
                       padding=True, truncation=True, max_length=512,
                       add_special_tokens=False, return_tensors="pt")

    cls_mask = inputs['input_ids'] == cls_token
    sep_mask = inputs['input_ids'] == sep_token

    num_of_seps = sep_mask.sum(1)

    # remove the last cls token if there are more cls tokens than sep tokens
    for row, seps in zip(cls_mask, num_of_seps):
        true_indices = row.nonzero().squeeze()

        if len(true_indices) > seps:
            last_true_index = true_indices[-1].item()
            row[last_true_index] = False

    num_of_clss = cls_mask.sum(1)
    df_chunk['num_of_cls'] = num_of_clss.tolist()

    df_chunk['remaining_labels'] = df_chunk.apply(lambda x: x['labels'][:x['num_of_cls']], axis=1)
    df_chunk['remaining_sent_labels'] = df_chunk.apply(lambda x: x['sent-labels'][:x['num_of_cls']], axis=1)

    outputs = model(**inputs.to(device))

    cls_vectors = outputs.last_hidden_state[cls_mask]

    return pd.DataFrame({'vectors': cls_vectors.detach().cpu().tolist(),
                         'labels': [item for sublist in df_chunk['remaining_labels'].tolist() for item in sublist],
                         'sent_labels': [item for sublist in df_chunk['remaining_sent_labels'].tolist() for item in
                                         sublist]
                         })


if __name__ == '__main__':
    main()
