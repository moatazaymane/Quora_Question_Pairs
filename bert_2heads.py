import torch
from torch import nn
import transformers
import config


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert1 = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert2 = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_dropout = nn.Dropout(0.2)
        self.output = nn.Linear(768, 1)

    def forward(self, ids1, ids2, token_type_ids1, token_type_ids2, mask1, mask2):
        _, b1 = self.bert1(ids=ids1,token_type_ids=token_type_ids1,attention_mask=mask1)
        _, b2 = self.bert2(ids=ids2, token_type_ids=token_type_ids2, attention_mask=mask2)



