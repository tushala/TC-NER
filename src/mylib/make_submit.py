# -*- coding: utf-8 -*-
import torch
import os
from src.config import *
from src.mylib import *
from src.data_proc.infer_cv import model_vote_mix

model_dict = {"idcnn": IDCNN_NER, "bilstm": BiLstm_Ner}
ner_data = open(statist_data_path, encoding="utf-8").read().split("\n")
ner_data = set([i.split(" ")[0] for i in ner_data])


def pick(sentence, pre_list, start_idx, count):
    # todo 先走个捷径
    res = []
    has_start = False
    word, symbol = "", ""
    last_pos = 0
    for i, (t, _tag) in enumerate(zip(sentence, pre_list)):
        tag = index2tag[_tag]
        if tag != O:
            if _tag % 2 == 1:
                if word:
                    word_length = len(word)
                    word = word.lstrip()
                    left_space = word_length - len(word)
                    word = word.rstrip()
                    if word in ner_data:
                        res.append(
                            f"T{count}\t{symbol} {last_pos+left_space} {last_pos+left_space+len(word)}\t{word}\n")
                        count += 1
                    word = ""
                last_pos = start_idx + i
                has_start = True
                word += t
                symbol = tag[2:]
            else:
                if has_start:
                    word += t
                else:
                    word = ""
                    has_start = False
    if word in ner_data:
        word_length = len(word)
        word = word.lstrip()
        left_space = word_length - len(word)
        word = word.rstrip()
        if word:
            res.append(f"T{count}\t{symbol} {last_pos+left_space} {last_pos+left_space+len(word)}\t{word}\n")
            count += 1
    return res, count


def make_submit(model_list):
    # todo
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(Vocab_Path)

    def _get_ner(content):
        nonlocal tokenizer
        # todo start ???
        res = []
        start = 0
        count = 1
        for idx, w in enumerate(content):
            if w in ['。', '！', '？', '?'] or idx == len(content) - 1:
                sentence = content[start:idx + 1]
                inputs = [tokenizer.convert_tokens_to_ids(i) for i in sentence]
                inputs = torch.tensor(inputs).unsqueeze(0)
                pre_list = model_vote_mix(model_states, inputs)
                pre_list = deocde_change(pre_list)
                _res, count = pick(sentence, pre_list, start, count)
                res.extend(_res)
                start = idx + 1
        return res

    def _submit(path):
        ann_num = os.path.split(path)[-1].split(".")[0]
        ann_path = submit_save_path + "/" + ann_num + ".ann"
        content = open(path, encoding="utf-8").read()
        # content = " 内分泌失调"
        res = _get_ner(content)

        with open(ann_path, 'w', encoding="utf-8") as f:
            for r in res:
                f.write(r)

    model_states = []
    for m_l in model_list:
        ml = os.path.split(m_l)[1]
        model_name = ml.split("_")[2]
        layer_idx = ml.index("layer_")
        start = layer_idx + len("layer_")
        layer_num = int(ml[start:start + 1])
        model = model_dict[model_name](pretrain_txt_roberta, last_n_layer=layer_num)
        # print(model)
        state = torch.load(m_l)
        model.load_state_dict(state)
        model_states.append(model)

    for path in os.listdir(submit_data_path):
        path = submit_data_path + '/' + path
        # path = submit_data_path + '/' + '1401.txt'
        _submit(path)
        print(f"{path} 提取完毕")
        # break


if __name__ == '__main__':
    model_list = [
        # SAVE_MODEL_DIR+"/ad0_cv2_bilstm_layer_4_0.7224.pth",
        # SAVE_MODEL_DIR+"/ad0_cv1_bilstm_layer_4_0.7119.pth",
        SAVE_MODEL_DIR + "/ad0_cv6_bilstm_layer_4_0.7345.pth",
        # SAVE_MODEL_DIR+"/ad0_cv5_bilstm_layer_4_0.7137.pth",
    ]
    # make_submit([SAVE_MODEL_DIR + '/ad0_cv3_bilstm_layer_4_0.7328.pth'])
    make_submit(model_list)
