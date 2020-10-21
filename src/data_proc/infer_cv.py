# -*- coding: utf-8 -*-
# cv训练
from transformers import BertTokenizer
from torch import nn
import torch
import pickle

from src.data_proc import build_dataloader, get_all_data
from src.data_proc.train_utils import main_train_cv
from src.data_proc import Text_Example
from src.model import *
from src.mylib import viterbi_decode, deocde_change

model_dict = {"idcnn": IDCNN_NER, "bilstm": BiLstm_Ner}


def train_cv(ad_train=False, last_n_layer=1):
    """7 折训练 f1大于0.69的保存"""
    assert Text_Example
    tokenizer = BertTokenizer.from_pretrained(Vocab_Path)
    if not os.path.exists(pickle_all_train_data_path):
        ner_data = get_all_data(train_data_path + txt_path, tokenizer, MAX_LENGTH)
    else:
        ner_data = pickle.load(open(pickle_all_train_data_path, 'rb'))
    # ner_data = pickle.load(open(pickle_all_aug_train_data_path, 'rb'))
    total_length = len(ner_data)
    for k in range(k_folds):
        model = BiLstm_Ner(pretrain_txt_roberta, last_n_layer=last_n_layer)
        # model = IDCNN_NER(pretrain_txt_roberta)
        train_idx = set(i for i in range(total_length) if i % k_folds != k)
        dev_idx = set(i for i in range(total_length) if i % k_folds == k)
        train_dataloader = build_dataloader(ner_data, train_idx, tokenizer, batch_size=BATCH_SIZE)
        dev_dataloader = build_dataloader(ner_data, dev_idx, tokenizer, batch_size=1)
        main_train_cv(k, model, train_dataloader, dev_dataloader, ad_train)
        # break


def weight_f1(t, lamb=1 / 4):
    import math
    '''
    # 牛顿冷却定律 
    '''
    return math.exp(-lamb * t)


def model_mix(model_list, inputs):
    """权重融合"""
    # cv3_bilstm_0.7166.pth
    emissions = None
    trans = None
    start_trans = None
    end_trans = None
    model_list = sorted(model_list, key=lambda x: float(x[-10:-4]), reverse=True)
    weights = [weight_f1(t) for t in range(len(model_list))]
    weight_sum = sum(weights)
    for ml, weight in zip(model_list, weights):
        model_name = ml.split("_")[2]
        layer_idx = ml.index("layer_")
        start = layer_idx + len("layer_")
        layer_num = int(ml[start:start + 1])
        model = model_dict[model_name](pretrain_txt_roberta, last_n_layer=layer_num)
        state = torch.load(SAVE_MODEL_DIR + ml)
        model.load_state_dict(state)
        model_ouput = model.get_output(inputs)
        if trans is None:
            trans = weight * model.crf.transitions
        else:
            trans += weight * model.crf.transitions
        if start_trans is None:
            start_trans = weight * model.crf.start_transitions
        else:
            start_trans += weight * model.crf.start_transitions
        if end_trans is None:
            end_trans = weight * model.crf.end_transitions
        else:
            end_trans += weight * model.crf.end_transitions
        if emissions is None:
            emissions = weight * model_ouput
        else:
            emissions += weight * model_ouput
    emissions /= weight_sum
    trans /= weight_sum
    start_trans /= weight_sum
    end_trans /= weight_sum
    res_tag_list = viterbi_decode(emissions, trans, start_trans, end_trans)[0]
    return res_tag_list


def model_vote_mix(model_list, inputs):
    """投票融合"""
    from collections import Counter
    all_ouputs = []
    vote_length = len(model_list)
    for ms in model_list:
        dec_res = ms.decode(inputs)[0]
        all_ouputs.append(dec_res)

    res = []
    for i in range(len(inputs[0])):
        vote_r = [all_ouputs[j][i] for j in range(vote_length)]
        # print(vote_r)
        vote_r, _ = Counter(vote_r).most_common(1)[0]
        res.append(vote_r)
    # res = deocde_change(res)
    return res


if __name__ == '__main__':
    model_list = [
        "/ad0_cv0_bilstm_layer_4_0.7242.pth",
        "/ad0_cv1_bilstm_layer_4_0.7119.pth",
        "/ad0_cv3_bilstm_layer_4_0.7328.pth",
        "/ad0_cv5_bilstm_layer_4_0.7137.pth",
    ]
    data = pickle.load(open(pickle_all_train_data_path, 'rb'))
    test_tokens = data[100].token_ids
    c_tags = data[100].tag_ids
    test_tokens = torch.tensor(test_tokens).unsqueeze(0).long()
    # pre = model_mix(model_list, test_tokens)
    pre = model_vote_mix(model_list, test_tokens)
    print(pre)
    # print(c_tags)
    # print(data[0].token_ids)
    w = [0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 8, 0, 5, 6, 6, 6, 25, 26, 26, 26, 0, 0, 0, 7, 8, 8, 8, 0, 5, 6, 6, 6, 0, 5, 6, 6, 6, 0, 0, 6, 0, 5, 6, 6, 6, 17, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 18, 17, 18, 18, 18, 18, 0, 0, 0, 0, 5, 6, 5, 6, 0, 5, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 22, 0, 23, 24, 0, 23, 24, 0, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 8, 0, 5, 6, 6, 6, 0, 5, 6, 6, 6, 0, 0, 6, 0, 5, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(w)
    w = deocde_change(w)
    print(w)