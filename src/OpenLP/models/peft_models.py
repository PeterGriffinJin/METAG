import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForSequenceClassification

class BertPEFTForLinkPredict(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, task_ids=None, **kwargs):

        node_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)

        return node_embeddings
