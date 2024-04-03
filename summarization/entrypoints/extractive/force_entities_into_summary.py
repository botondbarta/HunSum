import glob
import json

import click
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
import huspacy
import pandas as pd


def generate_summary_with_forced_entities(model, tokenizer, spacy, article, extractive_summary):
    named_entities = [ent.lemma_.replace('\\n\\n', ' ').replace('\\n', '').replace('" "', '').replace('""', '')
                      for ent in spacy(extractive_summary).ents
                      ]
    named_entities = list(set(named_entities))
    forced_tokens = []
    for named_entity in named_entities:
        forced_tokens.append(tokenizer.encode(named_entity, add_special_tokens=False))

    inputs = tokenizer.encode(article, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    if forced_tokens:
        outputs = model.generate(inputs.to('cuda'),
                                 force_words_ids=forced_tokens,
                                 no_repeat_ngram_size=3,
                                 num_beams=55,
                                 max_length=128,
                                 length_penalty=2,
                                 encoder_no_repeat_ngram_size=4,
                                 early_stopping=True)
    else:
        outputs = model.generate(inputs.to('cuda'),
                                 no_repeat_ngram_size=3,
                                 num_beams=55,
                                 max_length=128,
                                 length_penalty=2,
                                 encoder_no_repeat_ngram_size=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), named_entities


@click.command()
@click.argument('model_name')
@click.argument('model_path')
@click.argument('test_dir')
@click.argument('extractive_summaries')
@click.argument('generate_file')
def main(model_name, model_path, test_dir, extractive_summaries, generate_file):
    if model_name == 'mT5':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
    elif model_name == 'bert2bert':
        tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
        model = EncoderDecoderModel.from_pretrained(model_path).to('cuda')
    else:
        raise Exception('Unknown model name')

    nlp = huspacy.load('hu_core_news_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", ])

    files = sorted(glob.glob(f'{test_dir}/*.jsonl.gz'))
    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df = site_df[['article', 'uuid']]
        site_df = site_df.astype('str')
        site_dfs.append(site_df)
    test_df = pd.concat(site_dfs)
    extractive_df = pd.read_json(extractive_summaries, lines=True)
    df = pd.merge(test_df, extractive_df, on='uuid')
    generated_df = pd.DataFrame(columns=['uuid', 'generated_lead', 'named_entities'])
    for index, row in df.iterrows():
        article = row['article']
        extractive_summary = row['generated_lead']
        generated_summary, named_entities = generate_summary_with_forced_entities(model, tokenizer, nlp, article,
                                                                                  extractive_summary)
        generated_df = pd.concat([generated_df,
                                  pd.DataFrame(
                                      {
                                          'uuid': row['uuid'],
                                          'generated_lead': generated_summary,
                                          'named_entities': ','.join(named_entities),
                                      }, index=[0])])

        with open(generate_file, 'w', encoding='utf-8') as file:
            generated_df.to_json(file, force_ascii=False, lines=True, orient='records')


if '__main__' == __name__:
    main()
