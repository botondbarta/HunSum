import os
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from summarization.models.classifier import BertSummarizer

global_tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')


class ExtractiveDataset(Dataset):
    def __init__(self, dir_path, label_column):
        self.data = []
        dir_path = Path(dir_path)

        for file in os.listdir(dir_path):
            for chunk in pd.read_json(dir_path / file, lines=True, chunksize=200):
                chunk = chunk[['tokenizer_input', label_column]]
                chunk.rename(columns={label_column: 'label'}, inplace=True)
                self.data.extend(chunk.to_dict('records'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y = torch.tensor(sample['label'], dtype=torch.float32)
        return sample['tokenizer_input'], y


def collate(list_of_samples):
    src_sentences = [x[0] for x in list_of_samples]
    inputs = global_tokenizer(src_sentences,
                              padding=True, truncation=True, max_length=512,
                              add_special_tokens=False, return_tensors="pt")
    return inputs, torch.tensor([i for x in list_of_samples for i in x[1]], dtype=torch.float32)


@click.command()
@click.argument('train_dir')
@click.argument('valid_dir')
@click.argument('label_column')
@click.option('--batch_size', default=16, type=click.INT)
@click.option('--num_epochs', default=100, type=click.INT)
@click.option('--lr', default=5e-5, type=click.FLOAT)
def main(train_dir, valid_dir, label_column, batch_size, num_epochs, lr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset = ExtractiveDataset(train_dir, label_column)
    validset = ExtractiveDataset(valid_dir, label_column)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate)
    validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate)

    model = BertSummarizer()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch}')

        train_loss = []
        for i, (train_inputs, train_y) in enumerate(trainloader):
            input_ids = train_inputs['input_ids'].to(device)
            token_type_ids = train_inputs['token_type_ids'].to(device)
            attention_mask = train_inputs['attention_mask'].to(device)
            train_y = train_y.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, token_type_ids, attention_mask)

            loss = F.binary_cross_entropy(outputs.flatten(), train_y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

        with torch.no_grad():
            model.eval()
            valid_loss = []
            tp, tn, fp, fn = 0, 0, 0, 0
            for i, (valid_inputs, valid_y) in enumerate(validloader):
                input_ids = valid_inputs['input_ids'].to(device)
                token_type_ids = valid_inputs['token_type_ids'].to(device)
                attention_mask = valid_inputs['attention_mask'].to(device)
                valid_y = valid_y.to(device)

                outputs = model(input_ids, token_type_ids, attention_mask)
                outputs = outputs.flatten()

                loss = F.binary_cross_entropy(outputs, valid_y)
                valid_loss.append(loss.item())

                outputs = torch.round(outputs)

                tp += torch.sum(outputs * valid_y)
                tn += torch.sum((1 - outputs) * (1 - valid_y))
                fp += torch.sum(outputs * (1 - valid_y))
                fn += torch.sum((1 - outputs) * valid_y)

        print(f'Train loss:\t{sum(train_loss) / len(train_loss)}')
        print(f'Valid loss:\t{sum(valid_loss) / len(valid_loss)}')
        print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')


if __name__ == '__main__':
    main()
