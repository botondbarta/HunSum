import glob
import os.path

import click

from summarization.models.bert2bert import Bert2Bert
from summarization.models.mt5 import MT5


@click.command()
@click.argument('model_type')
@click.argument('article_path')
@click.argument('lead_dir')
@click.argument('config_path')
def main(model_type, article_path, lead_dir, config_path):
    if model_type == 'mt5':
        model = MT5(config_path)
    else:
        model = Bert2Bert(config_path)
    article_files = [article_path] if os.path.isfile(article_path) else glob.glob(f'{article_path}/*')
    for article_file in article_files:
        with open(article_file) as f:
            article = f.read()
        lead = model.predict_pipeline(article)
        lead_file = os.path.join(lead_dir, f'{os.path.basename(article_file).rsplit(".", 1)[0]}_lead.txt')
        with open(lead_file, 'w+') as f:
            f.write(lead[0]['summary_text'])


if __name__ == '__main__':
    main()
