import glob
import json
import os

import click
import datasets
import huspacy
import pandas as pd
from transformers import AutoTokenizer


@click.command()
@click.argument('src_dir')
@click.argument('label_column')
@click.argument('results_file')
@click.option('--use_stemming', is_flag=True, default=False)
def main(src_dir, label_column, results_file, use_stemming):
    tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
    rouge = datasets.load_metric("rouge")
    bert_score = datasets.load_metric("bertscore")

    files = sorted(glob.glob(f'{src_dir}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df = site_df[['lead', label_column]]
        site_df['tokenized'] = site_df[label_column].apply(lambda x: tokenizer(x,
                                                                                 padding=True, truncation=True,
                                                                                 max_length=510,
                                                                                 add_special_tokens=False,
                                                                                 return_tensors="pt"
                                                                                 )['input_ids'])
        site_df['truncated'] = site_df['tokenized'].apply(lambda x: tokenizer.batch_decode(x,
                                                          skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
        site_df = site_df[['lead', 'truncated']]
        site_df = site_df.astype('str')
        site_dfs.append(site_df)
    df = pd.concat(site_dfs)

    ref = df['lead'].tolist()
    gen = df['truncated'].tolist()

    bert_scores = bert_score.compute(predictions=gen, references=ref, model_type="SZTAKI-HLT/hubert-base-cc",
                                     num_layers=8, verbose=True)

    if use_stemming:
        nlp = huspacy.load()
        ref = [' '.join([token.lemma_ for token in nlp(r)]) for r in ref]
        gen = [' '.join([token.lemma_ for token in nlp(r)]) for r in gen]

    rouge_output = rouge.compute(
        predictions=gen, references=ref, rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    rouge1 = rouge_output["rouge1"].mid
    rouge2 = rouge_output["rouge2"].mid
    rougeL = rouge_output["rougeL"].mid

    avg = lambda x: sum(x) / len(x)

    scores = {
        "rouge1_precision": round(rouge1.precision, 4),
        "rouge1_recall": round(rouge1.recall, 4),
        "rouge1_fmeasure": round(rouge1.fmeasure, 4),
        "rouge2_precision": round(rouge2.precision, 4),
        "rouge2_recall": round(rouge2.recall, 4),
        "rouge2_fmeasure": round(rouge2.fmeasure, 4),
        "rougeL_precision": round(rougeL.precision, 4),
        "rougeL_recall": round(rougeL.recall, 4),
        "rougeL_fmeasure": round(rougeL.fmeasure, 4),
        "bert_score_precision": round(avg(bert_scores['precision']), 4),
        "bert_score_recall": round(avg(bert_scores['recall']), 4),
        "bert_score_f1": round(avg(bert_scores['f1']), 4),
    }

    with open(results_file, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    main()
