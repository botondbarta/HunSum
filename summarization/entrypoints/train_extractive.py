import click
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    #validset = MyDataset(valid_dir, label_column)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    #validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Classifier(768, 50, 0.1)
    model.to(device)

    #criterion = nn.BCELoss()
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
            weights = train_y*9+1

            optimizer.zero_grad()
            outputs = model(train_x)

            loss = nn.BCELoss(weight=weights)(outputs[:, 0], train_y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        print(f'Train loss:\t{sum(train_loss)/len(train_loss)}')


if __name__ == '__main__':
    main()
