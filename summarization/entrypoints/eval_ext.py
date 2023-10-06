import json
import os
from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from summarization.models.classifier import BertSummarizer

global_tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')


class ExtractiveDataset3(Dataset):
    def __init__(self, dir_path, label_column):
        self.data = []
        dir_path = Path(dir_path)

        for file in os.listdir(dir_path):
            df = pd.read_json(dir_path / file, lines=True)
            df.rename(columns={label_column: 'label'}, inplace=True)
            df = df[['uuid', 'tokenizer_input', 'label', 'tokenized_article']]
            self.data.extend(df.to_dict('records'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y = torch.tensor(sample['label'], dtype=torch.float32)
        return sample['tokenizer_input'], y, sample['uuid'], sample['tokenized_article']


def collate2(list_of_samples):
    src_sentences = [x[0].replace('[PAD]', '').strip() for x in list_of_samples]
    uuids = [x[2] for x in list_of_samples]
    tokenized_articles = [x[3] for x in list_of_samples]
    inputs = global_tokenizer(src_sentences,
                              padding=True, truncation=True, max_length=512,
                              add_special_tokens=False, return_tensors="pt")
    return inputs, torch.tensor([i for x in list_of_samples for i in x[1]], dtype=torch.float32), uuids, tokenized_articles


@click.command()
@click.argument('model_dir')
@click.argument('test_dir')
@click.argument('output_dir')
@click.argument('label_column')
def main(model_dir, test_dir, output_dir, label_column):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertSummarizer()
    model.load_state_dict(torch.load(Path(model_dir) / 'model.pt', map_location='cpu'))
    model.eval()

    testset = ExtractiveDataset3(test_dir, label_column)
    testloader = DataLoader(dataset=testset, batch_size=1, pin_memory=True, collate_fn=collate2)

    gen_leads = []

    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (test_input, test_y, uuid, tokenized_article) in enumerate(testloader):
        input_ids = test_input['input_ids'].to(device)
        token_type_ids = test_input['token_type_ids'].to(device)
        attention_mask = test_input['attention_mask'].to(device)
        test_y = test_y.to(device)

        outputs = model(input_ids, token_type_ids, attention_mask, global_tokenizer.cls_token_id)
        outputs = outputs.flatten()

        outputs[outputs.topk(3).indices] = 1
        outputs[outputs.topk(len(outputs) - 3, largest=False).indices] = 0

        tp += torch.sum(outputs * test_y)
        tn += torch.sum((1 - outputs) * (1 - test_y))
        fp += torch.sum(outputs * (1 - test_y))
        fn += torch.sum((1 - outputs) * test_y)
        gen_lead = ' '.join([tokenized_article[0][i] for i, number in enumerate(outputs.to(torch.int)) if number == 1])
        gen_leads.append({'uuid': uuid[0], 'generated_lead': gen_lead})

    lead_file = os.path.join(Path(output_dir) / 'generated_leads.jsonl')
    result_file = os.path.join(Path(output_dir) / 'results.json')
    results = {
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1': 2 * tp / (2 * tp + fp + fn),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }
    df = pd.DataFrame(gen_leads)
    with open(result_file, 'w') as file:
        json.dump(results, file)
    with open(lead_file, 'w', encoding='utf-8') as file:
        df.to_json(file, force_ascii=False, lines=True, orient='records')


if __name__ == '__main__':
    main()
