import glob
import json
import os

import click
import datasets
import huspacy
import pandas as pd


@click.command()
@click.argument('references')  # path to original json file(s)
@click.argument('predicted')  # path to predicted json file
@click.argument('results_file')
@click.option('--use_stemming', is_flag=True, default=False)
def main(references, predicted, results_file, use_stemming):
    rouge = datasets.load_metric("rouge")
    bert_score = datasets.load_metric("bertscore")
    # rouge = evaluate.load("rouge")

    files = [references] if os.path.isfile(references) else sorted(glob.glob(f'{references}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df = site_df[['lead', 'article', 'uuid']]
        site_df = site_df.astype('str')
        site_dfs.append(site_df)
    ref_df = pd.concat(site_dfs)
    pred_df = pd.read_json(predicted, lines=True)

    df = pd.merge(ref_df, pred_df, on='uuid')

    ref = df['lead'].tolist()
    gen = df['generated_lead'].tolist()

    if use_stemming:
        nlp = huspacy.load()
        ref = [' '.join([token.lemma_ for token in nlp(r)]) for r in ref]
        gen = [' '.join([token.lemma_ for token in nlp(r)]) for r in gen]

    rouge_output = rouge.compute(
        predictions=gen, references=ref, rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    bert_scores = bert_score.compute(predictions=gen, references=ref, model_type="SZTAKI-HLT/hubert-base-cc",
                                     num_layers=8)

    rouge1 = rouge_output["rouge1"].mid
    rouge2 = rouge_output["rouge2"].mid
    rougeL = rouge_output["rougeL"].mid

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
        "bert_score_precision": round(bert_scores['precision'], 4),
        "bert_score_recall": round(bert_scores['recall'], 4),
        "bert_score_f1": round(bert_scores['f1'], 4),
    }

    with open(results_file, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    main()
