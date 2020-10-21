# -*- coding: utf-8 -*-
from src.pretrain.create_pretrain_data import main_
from src.config import *
from tqdm import tqdm, trange
from transformers import BertForMaskedLM, AdamW
import torch
from src.mylib import logger


def do_pretrain_bert():
    logger.info("pretrain start")
    model = BertForMaskedLM.from_pretrained(pretrain_roberta)
    model = model.to(DEVICE)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    train_dataloader = main_([train_data_path + txt_path, test_data_path + txt_path])
    model.train()
    for e in range(train_epoch):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _, _ = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                               masked_lm_labels=label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch: {e} step: {step} loss : {loss.tolist()}")
        logger.info(f"epoch: {e} loss : {loss.tolist()}")
        if e > 4:
            save_path = pretrain_save_path + f"{e}.bin"
            torch.save(model.state_dict(), save_path)
    logger.info("pretrain finish")


if __name__ == '__main__':
    do_pretrain_bert()
