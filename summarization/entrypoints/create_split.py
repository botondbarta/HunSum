import glob
import os
import subprocess

import click
import numpy as np
import pandas as pd

from summarization.utils.data_helpers import make_dir_if_not_exists, get_domain_of_df_site


@click.command()
@click.argument('src_dir')
@click.argument('out_dir')
@click.argument('test_dev_size', type=click.INT)
def main(src_dir, out_dir, test_dev_size):
    make_dir_if_not_exists(out_dir)
    sites = glob.glob(f'{src_dir}/*.jsonl.gz')

    total_size = 0
    for site in sites:
        command = f'zcat {site} | wc -l'
        total_size += int(run_linux_command(command))
    for site in sites:
        split_and_save_site(site, total_size, test_dev_size, out_dir)


def split_and_save_site(site, total_size, test_dev_size, out_dir):
    df_site = pd.read_json(site, lines=True)
    domain = get_domain_of_df_site(df_site)
    df_size = len(df_site)
    valid_size = (df_size / total_size) * test_dev_size
    train_size = df_size - 2 * valid_size
    train, valid, test = np.split(df_site.sample(frac=1, random_state=123),
                                  [train_size, train_size + valid_size])

    make_dir_if_not_exists(os.path.join(out_dir, 'train'))
    make_dir_if_not_exists(os.path.join(out_dir, 'valid'))
    make_dir_if_not_exists(os.path.join(out_dir, 'test'))
    train.to_json(f'{out_dir}/train/{domain}_train.jsonl.gz', orient='records', lines=True, compression='gzip')
    valid.to_json(f'{out_dir}/valid/{domain}_valid.jsonl.gz', orient='records', lines=True, compression='gzip')
    test.to_json(f'{out_dir}/test/{domain}_test.jsonl.gz', orient='records', lines=True, compression='gzip')


def run_linux_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        # If there's an error, you can handle it as needed.
        print(f"Error occurred: {result.stderr.strip()}")
        raise SystemError('Error occurred while running command')


if __name__ == '__main__':
    main()
