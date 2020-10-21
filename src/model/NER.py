# -*- coding: utf-8 -*-
from src.model.crf import CRF
from transformers import BertModel
from torch import nn
from src.mylib.const import tag2indx
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from src.config import *


class BaseNer(nn.Module):
    MODEL_NAME = 'base'

    def __init__(self, pretrain_path, last_n_layer=1, emb_size=EMB_SIZE, batch_first=True):
        super(BaseNer, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.hidden_size = self.bert.config.hidden_size
        self.emb = emb_size
        self.last_n_layer = last_n_layer
        self.tag_linear = nn.Linear(emb_size * 2, len(tag2indx))
        self.crf = CRF(len(tag2indx), batch_first)
        self.last_n_layer = last_n_layer
        self.weight = nn.Parameter(torch.rand(last_n_layer))

    def forward(self, inputs, tags, mask, reduction="mean"):
        outputs = self.get_output(inputs, mask)
        score = -self.crf(outputs, tags=tags, mask=mask, reduction=reduction)
        return score

    def get_bert_output(self, inputs, mask):
        if self.last_n_layer == 1:
            bert_outputs, _, _ = self.bert(inputs, attention_mask=mask)
        else:
            _, _, hidden_states = self.bert(inputs, attention_mask=mask)
            bert_outputs = torch.zeros(hidden_states[-1].size()).to(self.bert.device)
            # weight_sum = torch.sum(self.weight, dim=0)
            layer_weight = F.softmax(self.weight, dim=0)
            for layer_output, weight in zip(hidden_states[-self.last_n_layer:], layer_weight):
                bert_outputs += layer_output * weight
        bert_outputs = F.dropout(bert_outputs, p=0.1)
        return bert_outputs

    def get_output(self, inputs, mask=None):
        bert_outputs = self.get_bert_output(inputs, mask)
        outputs = self.tag_linear(bert_outputs)
        return outputs

    def decode(self, inputs, mask=None):
        outputs = self.get_output(inputs, mask)
        best_tags_list = self.crf.decode(outputs, mask=mask)
        return best_tags_list


class BiLstm_Ner(BaseNer):
    MODEL_NAME = 'bilstm'

    def __init__(self, pretrain_path, last_n_layer=1, emb_size=EMB_SIZE):
        super(BiLstm_Ner, self).__init__(pretrain_path, last_n_layer, emb_size)
        self.lstm = nn.LSTM(self.hidden_size, emb_size, bidirectional=True, num_layers=1)

    def get_output(self, inputs, mask=None):
        bert_outputs = self.get_bert_output(inputs, mask)
        init_hidden = self._init_hidden(bert_outputs.size(1), self.emb)
        lstm_output, _ = self.lstm(bert_outputs, init_hidden)

        outputs = self.tag_linear(lstm_output)
        return outputs

    def _init_hidden(self, max_length, emb):
        hidden = Variable(torch.randn(2, max_length, emb).to(self.bert.device)), \
                 Variable(torch.randn(2, max_length, emb).to(self.bert.device))
        return hidden


# class IDCNN_NER(BaseNer):
#     MODEL_NAME = 'idcnn'
#
#     def __init__(self, pretrain_path, emb_size=EMB_SIZE):
#         super(IDCNN_NER, self).__init__(pretrain_path, emb_size)
#         cnn_layer = [1, 1, 2]
#         self.idcnn = nn.ModuleList(
#             nn.Conv2d(1, MAX_LENGTH, 2 * i, stride=768 // 1, dilation=i) for i in cnn_layer
#         )
#         self.linear = nn.Linear(len(cnn_layer), len(tag2indx))
#
#     def get_output(self, inputs, mask):
#         bert_outputs = self.get_bert_output(inputs, mask)
#         bert_outputs = bert_outputs.unsqueeze(1)
#         cnn_output = [cnn(bert_outputs).transpose_(2, 1) for cnn in self.idcnn]
#         cnn_output = torch.cat(cnn_output, dim=-1)
#         output = self.linear(cnn_output)
#         output = output.squeeze(1)
#         return output

class IDCNN_NER(BaseNer):
    MODEL_NAME = 'idcnn'

    def __init__(self, pretrain_path, last_n_layer=1, emb_size=EMB_SIZE, kernel_size=3, num_block=4):
        super(IDCNN_NER, self).__init__(pretrain_path, last_n_layer, emb_size)
        cnn_layer = [1, 1, 2]
        output_size = len(tag2indx)
        hidden_size = self.bert.config.hidden_size
        net = nn.Sequential()
        for i in range(len(cnn_layer)):
            dilation = cnn_layer[i]
            single_block = nn.Conv1d(in_channels=MAX_LENGTH,
                                     out_channels=MAX_LENGTH,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)

            net.add_module(f"layer{i}", single_block)
            net.add_module(f"relu", nn.ReLU())
            net.add_module(f"layernorm", nn.LayerNorm(hidden_size))

        self.linear = nn.Linear(hidden_size, output_size)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module(f"block{i}", net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", nn.LayerNorm(hidden_size))

    def get_output(self, inputs, mask=None):
        bert_outputs = self.get_bert_output(inputs, mask)
        output = self.idcnn(bert_outputs)
        output = self.linear(output)
        return output
