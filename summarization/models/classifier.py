import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=100, dropout=0.2):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)
