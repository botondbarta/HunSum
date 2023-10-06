import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=100, dropout=0.2, output_dim=1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)


class BertSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
        self.bert = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')
        self.classifier = Classifier(self.bert.config.hidden_size, 50, 0.1, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        cls_mask = input_ids == self.tokenizer.cls_token_id
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[cls_mask])
