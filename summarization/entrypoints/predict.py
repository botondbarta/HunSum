import glob
import os.path

import click
import pandas as pd

from summarization.models.bert2bert import Bert2Bert
from summarization.models.mt5 import MT5


@click.command()
@click.argument('model_type')
@click.argument('article_path')
@click.argument('lead_dir')
@click.argument('config_path')
@click.option('--use_jsonl', default=False)
def main(model_type, article_path, lead_dir, config_path, use_jsonl):
    if model_type == 'mt5':
        model = MT5(config_path)
    else:
        model = Bert2Bert(config_path)
    article_files = [article_path] if os.path.isfile(article_path) else glob.glob(f'{article_path}/*')
    if not use_jsonl:
        for article_file in article_files:
            with open(article_file) as f:
                article = f.read()
            lead = model.predict_pipeline(article)
            lead_file = os.path.join(lead_dir, f'{os.path.basename(article_file).rsplit(".", 1)[0]}_lead.txt')
            with open(lead_file, 'w+') as f:
                f.write(lead[0]['summary_text'])
    else:
        dfs = []
        for article_file in article_files:
            df = pd.read_json(article_file, lines=True)
            leads = model.predict_pipeline(df.article.tolist())
            df['generated_lead'] = [lead['summary_text'] for lead in leads]
            df = df[['article', 'lead', 'generated_lead', 'uuid']]
            dfs.append(df)

        df = pd.concat(dfs, axis=1)
        lead_file = os.path.join(lead_dir, f'generated_leads.jsonl')
        with open(lead_file, 'w', encoding='utf-8') as file:
            df.to_json(file, force_ascii=False, lines=True, orient='records')




if __name__ == '__main__':
    main()
