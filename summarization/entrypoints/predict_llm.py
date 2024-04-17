import glob

import click
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM

tqdm.pandas()


def get_model_and_tokenizer(model_name, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = 'left'

    compute_dtype = "bfloat16"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map='auto')

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def get_test_files(path):
    files = sorted(glob.glob(f'{path}/*.jsonl.gz'))

    site_dfs = []
    for file in files:
        site_df = pd.read_json(file, lines=True)
        site_df = site_df[['lead', 'article', 'uuid']]
        site_df = site_df.astype('str')
        site_dfs.append(site_df)
    df = pd.concat(site_dfs)
    return df


def generate(model, tokenizer, pad_token_id, article: str):
    prompt = ('### Utasítás:\n'
              + 'Írj egy rövid összefoglalót a következő cikkről anélkül, hogy elveszítené az információk jelentőségét.'
              + "\n\n### Cikk:\n" + str(article)
              + "\n\n### Összefoglaló:\n")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_config = GenerationConfig(do_sample=True,
                                         num_beams=5,
                                         temperature=0.4,
                                         return_dict_in_generate=True,
                                         output_scores=True,
                                         pad_token_id=pad_token_id,
                                         max_new_tokens=128
                                         )

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config)

    for seq in generation_output.sequences:
        output = tokenizer.decode(seq, skip_special_tokens=True)
        return output


@click.command()
@click.argument('adapter_path')
@click.argument('test_dir')
@click.argument('out_dir')
def main(adapter_path, test_dir, out_dir):
    model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    model, tokenizer = get_model_and_tokenizer(model_name, adapter_path)

    pad_token_id = tokenizer.pad_token_id

    df = get_test_files(test_dir)
    df['generated_lead'] = df['article'].progress_apply(
        lambda x: generate(model, tokenizer, pad_token_id, x).split('Összefoglaló:\n')[1].strip())

    df = df[['uuid', 'generated_lead']]
    df.to_json(f'{out_dir}/generated_leads.jsonl', lines=True, orient='records', force_ascii=False)


if __name__ == '__main__':
    main()
