import torch.nn as nn
from transformers import AutoModel


class BertSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask, cls_id):
        cls_mask = input_ids == cls_id
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = self.linear(outputs.last_hidden_state[cls_mask])
        return self.sigmoid(x)
