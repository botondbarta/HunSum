import click
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from summarization.models.classifier import Classifier
from pathlib import Path


class ExtractiveDataset(Dataset):
    def __init__(self, dir_path, label_column):
        self.data = []
        dir_path = Path(dir_path)

        for file in os.listdir(dir_path):
            for chunk in pd.read_json(dir_path / file, lines=True, chunksize=1000):
                chunk = chunk[['vectors', label_column]]
                chunk.rename(columns={label_column: 'label'}, inplace=True)
                self.data.extend(chunk.to_dict('records'))
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = torch.tensor(sample['vectors'], dtype=torch.float32)
        y = torch.tensor(sample['label'], dtype=torch.float32)
        return x, y


@click.command()
@click.argument('train_dir')
@click.argument('valid_dir')
@click.argument('label_column')
@click.option('--batch_size', default=512, type=click.INT)
def main(train_dir, valid_dir, label_column, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset = ExtractiveDataset(train_dir, label_column)
    validset = ExtractiveDataset(valid_dir, label_column)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Classifier(768, 50, 0.1)
    model.to(device)

    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    num_epochs = 10000

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch}')

        train_loss = []
        for i, (train_x, train_y) in enumerate(trainloader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            optimizer.zero_grad()
            outputs = model(train_x)

            loss = F.binary_cross_entropy(outputs.flatten(), train_y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

        with torch.no_grad():
            model.eval()
            valid_loss = []
            tp, tn, fp, fn = 0, 0, 0, 0
            for i, (valid_x, valid_y) in enumerate(validloader):
                valid_x = valid_x.to(device)
                valid_y = valid_y.to(device)

                outputs = model(valid_x)
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
