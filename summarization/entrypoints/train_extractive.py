import os
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from summarization.models.bertsum import BertSum
from summarization.utils.logger import get_logger

global_tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')


class ExtractiveDataset(Dataset):
    def __init__(self, dir_path, label_column, drop=False):
        self.data = []
        dir_path = Path(dir_path)

        for file in os.listdir(dir_path):
            for chunk in pd.read_json(dir_path / file, lines=True, chunksize=10000):
                chunk.rename(columns={label_column: 'label'}, inplace=True)

                if drop:
                    chunk['sum'] = chunk['label'].apply(lambda x: sum(x))
                    chunk = chunk[chunk['sum'] > 0]

                chunk = chunk[['tokenizer_input', 'label']]
                self.data.extend(chunk.to_dict('records'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y = torch.tensor(sample['label'], dtype=torch.float32)
        return sample['tokenizer_input'], y


def collate(list_of_samples):
    src_sentences = [x[0] for x in list_of_samples]
    inputs = global_tokenizer([sent.replace('[PAD]', '').strip() for sent in src_sentences],
                              padding=True, truncation=True, max_length=512,
                              add_special_tokens=False, return_tensors="pt")
    return inputs, torch.tensor([i for x in list_of_samples for i in x[1]], dtype=torch.float32)


def validate(model, device, validloader):
    model.eval()
    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, (valid_inputs, valid_y) in enumerate(validloader):
            input_ids = valid_inputs['input_ids'].to(f'cuda:{model.device_ids[0]}')
            token_type_ids = valid_inputs['token_type_ids'].to(f'cuda:{model.device_ids[0]}')
            attention_mask = valid_inputs['attention_mask'].to(f'cuda:{model.device_ids[0]}')
            valid_y = valid_y.to(f'cuda:{model.device_ids[0]}')

            outputs = model(input_ids, token_type_ids, attention_mask, global_tokenizer.cls_token_id)
            outputs = outputs.flatten()

            loss = F.binary_cross_entropy(outputs, valid_y)

            top_k = int(sum(valid_y).item())
            outputs[outputs.topk(top_k).indices] = 1
            outputs[outputs.topk(len(outputs) - top_k, largest=False).indices] = 0

            tp += torch.sum(outputs * valid_y)
            tn += torch.sum((1 - outputs) * (1 - valid_y))
            fp += torch.sum(outputs * (1 - valid_y))
            fn += torch.sum((1 - outputs) * valid_y)
        return loss.item(), tp, tn, fp, fn


@click.command()
@click.argument('train_dir')
@click.argument('valid_dir')
@click.argument('model_dir')
@click.argument('label_column')
@click.option('--batch_size', default=16, type=click.INT)
@click.option('--num_epochs', default=100, type=click.INT)
@click.option('--lr', default=5e-5, type=click.FLOAT)
@click.option('--patience', default=5, type=click.INT)
@click.option('--validation_step', default=1, type=click.INT)
def main(train_dir, valid_dir, model_dir, label_column, batch_size, num_epochs, lr, patience, validation_step):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    early_stop = False
    logger = get_logger('logger', Path(model_dir) / 'log.txt')

    trainset = ExtractiveDataset(train_dir, label_column, drop=True)
    validset = ExtractiveDataset(valid_dir, label_column)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate)
    validloader = DataLoader(dataset=validset, batch_size=1, pin_memory=True, collate_fn=collate)

    model = BertSum()
    model = torch.nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    valid_loss = []
    recalls = []
    curr_patience = 0

    for epoch in range(num_epochs):
        if curr_patience == patience:
            break
        if early_stop:
            break
        model.train()
        logger.info(f'Epoch {epoch}')

        for i, (train_inputs, train_y) in tqdm(enumerate(trainloader)):
            if curr_patience == patience:
                break
            if early_stop:
                break
            input_ids = train_inputs['input_ids'].to(f'cuda:{model.device_ids[0]}')
            token_type_ids = train_inputs['token_type_ids'].to(f'cuda:{model.device_ids[0]}')
            attention_mask = train_inputs['attention_mask'].to(f'cuda:{model.device_ids[0]}')
            train_y = train_y.to(f'cuda:{model.device_ids[0]}')

            optimizer.zero_grad()
            outputs = model(input_ids, token_type_ids, attention_mask, global_tokenizer.cls_token_id)

            loss = F.binary_cross_entropy(outputs.flatten(), train_y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            if i % validation_step == 0:
                loss, tp, tn, fp, fn = validate(model, device, validloader)
                valid_loss.append(loss)
                logger.info(f'Train loss:\t{sum(train_loss[-1000:]) / 1000}')
                logger.info(f'Valid loss:\t{loss}')
                logger.info(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                logger.info(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
                recalls.append(recall)

                if len(valid_loss) == 1 or loss < min(valid_loss[:-1]):
                    logger.info('improved, saving')
                    torch.save(model.module.state_dict(), Path(model_dir) / 'model.pt')
                    curr_patience = 0
                else:
                    curr_patience += 1
                model.train()


if __name__ == '__main__':
    main()
