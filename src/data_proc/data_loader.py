# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from src.mylib import tag2indx, Vocab_Path, pickle_data_path
import glob
import os
from src.config import *
from transformers import BertTokenizer
from collections import namedtuple
from src.mylib.const import O
from src.model import *
from src.data_proc.data_utils import Text_Example


class SampleDataset(Dataset):
    def __init__(self, ner_data, index: set, tokenizer: BertTokenizer, max_len=MAX_LENGTH):
        c_data = [nd for i, nd in enumerate(ner_data) if i in index]
        self.token_ids = [sample.token_ids for sample in c_data]
        self.tags = [sample.tag_ids for sample in c_data]
        self.texts = [sample.text for sample in c_data]
        self.anns = [sample.anns for sample in c_data]
        self._len = len(c_data)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def pad_sent_ids(self, sent_ids, max_length, padded_token_id):
        mask = [1] * (min(len(sent_ids), max_length)) + [0] * (max_length - len(sent_ids))
        sent_ids = sent_ids[:max_length] + [padded_token_id] * (max_length - len(sent_ids))

        return sent_ids, mask

    def __getitem__(self, index):
        sent_token_ids = self.token_ids[index]
        tag_ids = self.tags[index]
        sent_token_ids, mask = self.pad_sent_ids(
            sent_token_ids, max_length=self.max_len, padded_token_id=self.tokenizer.pad_token_id)
        tag_ids, _ = self.pad_sent_ids(tag_ids, max_length=self.max_len, padded_token_id=tag2indx[O])
        return {"token_ids": torch.tensor(sent_token_ids, dtype=torch.long),
                "tags": torch.tensor(tag_ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "text": self.texts[index],
                "anns": self.anns[index]}

    def __len__(self):
        return self._len


def build_dataloader(ner_data, index: set, tokenizer: BertTokenizer, batch_size, shuffle=True):
    dataset = SampleDataset(ner_data, index, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# if __name__ == '__main__':
#     t = BertTokenizer.from_pretrained(Vocab_Path)
#     d = build_dataloader(pickle_all_train_data_path, set(range(200)), t, BATCH_SIZE)
#     for i in d:
#         # print(1234567, i["text"])
#         print(1234567, i["anns"])
#         break
