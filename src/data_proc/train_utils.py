# -*- coding: utf-8 -*-
import torch
from torch import nn
from src.data_proc import build_dataloader, Text_Example
from src.config import *
from transformers import BertTokenizer

from src.model import *
from src.ad_train import FGM

from tqdm import tqdm
from src.mylib import logger, load_data, index2tag, O, deocde_change


def train(model, optimizer, dataloader, epoch, device, ad_train):
    fgm = FGM(model)
    model.train()
    pbar = tqdm(dataloader)
    for data in pbar:
        inputs, tags, mask = data['token_ids'], data['tags'], data['mask'].bool()
        inputs = inputs.to(device)
        tags = tags.to(device)
        mask = mask.to(device)
        if ad_train:
            fgm.attack()
        loss = model(inputs, tags, mask)
        optimizer.zero_grad()
        loss.backward()
        if ad_train:
            fgm.restore()
        optimizer.step()
        pbar.set_description(f"Train epoch: {epoch}/{EPOCH}, loss: {loss.tolist():.3f}")
    logger.info(f"Train epoch: {epoch+1}/{EPOCH}, loss: {loss.tolist():.3f}")


def extract(text, tag_list):
    # 感觉解码还是有些问题
    assert len(text) == len(tag_list)
    res = []
    word, symbol = "", ""
    for i, (t, _tag) in enumerate(zip(text, tag_list)):
        tag = index2tag[_tag]
        if tag != O:
            if _tag % 2 == 1:
                if word:
                    res.append((symbol, word))
                word = t
                symbol = tag[2:]
            else:
                if word:
                    word += t
        else:
            if word:
                res.append((symbol, word))
                word, symbol = "", ""
    if word:
        res.append((symbol, word))
    return res


def make_optimizer(model: BaseNer):
    from torch.optim import Adam
    # for name, param in model.named_parameters():
    #     if "bert_module" in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    crf_parmas = list(map(id, model.crf.parameters()))
    base_params = filter(lambda p: id(p) not in crf_parmas and p.requires_grad, model.parameters())
    # optimizer = Adam([para for para in model.parameters() if para.requires_grad],
    #                  lr=LR, weight_decay=WEIGHT_DECAY)
    params = [
        {'params': base_params},
        {'params': model.crf.parameters(), "lr": LR * 8}  # crf 学习率设为 BERT 的 5～10
    ]
    optimizer = Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    return optimizer


def evaluate(model, dev_dataloader, epoch, device=DEVICE):
    model.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for data in dev_dataloader:
        inputs, mask, text, anns = \
            data['token_ids'], data['mask'].bool(), data['text'][0], data['anns']
        inputs = inputs.to(device)
        mask = mask.to(device)
        decode_list = model.decode(inputs, mask)[0]
        # decode_list = deocde_change(decode_list)

        R = set(extract(text, decode_list))  # 预测转换
        T = set([(i[0][0], i[1][0]) for i in anns])  # 真实
        # if epoch > 6:
        #     print(-1, text)
        #     print(0, decode_list)
        #     print(1, R)
        #     if any(i[0] == "" for i in R):
        #         logger.info(12321, text)
        #         logger.info(32123, decode_list)
        # print(2, T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall)
    logger.info(f"Eval epoch {epoch+1} f1: {f1}, precision: {precision}, recall:{recall}")
    return f1, precision, recall


def main_train_cv(cv: int, model: nn.Module, train_dataloader, dev_dataloader, ad_train):
    assert Text_Example
    optimizer = make_optimizer(model)
    best_evaluate_score = 0.
    model = model.to(DEVICE)
    for epoch in range(EPOCH):
        train(model, optimizer, train_dataloader, ad_train=ad_train, epoch=epoch, device=DEVICE)
        evaluate_score, _, _ = evaluate(model, dev_dataloader, epoch=epoch, device=DEVICE)
        if evaluate_score >= best_evaluate_score:
            best_evaluate_score = evaluate_score
        if evaluate_score > 0.7:
            save_model_path = os.path.join(SAVE_MODEL_DIR,
                                           f'ad{int(ad_train)}_cv{cv}_{model.MODEL_NAME}_layer_{model.last_n_layer}_{evaluate_score:.4f}.pth')  # todo
            logger.info(f'cv{cv} saving model to {save_model_path}, best_score={best_evaluate_score}')
            torch.save(model.cpu().state_dict(), save_model_path)
        model.to(DEVICE)
        print(f"epoch: {epoch} weight: {model.weight}")
    logger.info(f'cv{cv} best_score={best_evaluate_score}')
    # assert
