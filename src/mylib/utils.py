# -*- coding: utf-8 -*-
import glob
from sklearn.model_selection import train_test_split
from src.config import *
from src.mylib.const import O, index2tag
from src.model import *
import os
import re
from tqdm import tqdm
from transformers import BertTokenizer
import torch

space_compile = re.compile(r"\s")

test_conent = """每丸重9g  口服。一次1丸，一日3次  本品养血祛瘀，血热养血祛瘀，血热证者忌用  山西双人药业有限责任公司  1、.收缩子宫作用生化汤可增强兔离体和在体子宫平滑肌收缩的幅度和频率。正常产产妇及剖宫产产妇产后服用小剂量生化丸可促进子宫复原。2、促进造血作用生化汤灌胃能升高失血性小鼠的红细胞及血红蛋白含量。3、其他作用生化汤灌胃能抑制大鼠蛋清性足肿胀，抑制醋酸所致小鼠扭体反应。生化汤大鼠灌胃给药，能降低冰水加肾腺素所致急性“血瘀”模型大鼠的全血黏度，改善血液流变学。生化汤可抑制体外人血栓的形成。4、生化汤24.6g/kg灌胃给药14天，大鼠生长发育、体重、肝功能、肾功能、血常规及脏器未见有明显异常。  养血祛瘀。用于产后受寒恶露不行或行而不畅，夹有血块，小腹冷痛行而不畅恶露不行养血祛瘀 尚不明确。 """


def cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:
        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def make_new_data(data_path, save_path, text_length=MAX_LENGTH):
    q_dic = {}
    data_num = os.path.split(data_path)[-1].split(".")[0]
    txt_path = data_path
    ann_path = data_path.replace("txt", "ann")
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            _, sy_st_en, name = line.split("\t")
            sy, st, en = sy_st_en.split(" ")
            start_index, end_index = int(st), int(en)
            length = end_index - start_index
            for r in range(length):
                if r == 0:
                    q_dic[start_index] = f"B-{sy}"
                else:
                    q_dic[start_index + r] = f"I-{sy}"

    content_str = open(txt_path, encoding="utf-8").read()
    cut_text_list, cut_index_list = cut_test_set([content_str], text_length)
    start_ = 0

    for idx, line in enumerate(cut_text_list):
        write_path = save_path + data_num + f"_{idx}.txt"
        with open(write_path, "w", encoding="utf-8") as f:
            for str_ in line:
                if space_compile.match(str_):
                    pass
                else:
                    tag = q_dic.get(start_, O)
                    f.write(f"{str_} {tag}\n")
                start_ += 1
            f.write('%s\n' % "END O")


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    # print(c)
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


def show_test(input, tag):
    """
    检验切词是否正确，无卵用
    """
    t = BertTokenizer.from_pretrained(Vocab_Path)
    w = "".join(t.convert_ids_to_tokens(input))
    tags = [index2tag[i.tolist()] for i in tag]
    start = 0
    end = 0
    symbol = ""
    res = []
    start_set = set()
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            symbol = t[2:]
            start = i
        if t.startswith("I-"):
            end = i
        if t == 'O' and start and start not in start_set:
            start_set.add(start)
            res.append((symbol, w[start:end + 1]))
    print(res)


def tag_statist(path):
    '''统计各个实体的数量 用于数据增强'''
    from collections import defaultdict
    import pickle
    rec = defaultdict(dict)
    for path in glob.glob(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                line, word = line[1], line[-1]
                tag, _, _ = line.split(" ")
                _names = rec[tag].get(len(word), None)
                if _names is None:
                    r = {word}
                else:
                    _names.add(word)
                    r = _names
                rec[tag][len(word)] = r
    with open(statist_entity_data_path, "wb") as f:
        pickle.dump(rec, f)
    # res = sorted(res.items(), key=lambda x: -x[1])
    # with open(statist_entity_data_path, 'w', encoding="utf-8") as f:
    #     for (k, v) in res:
    #         f.write(str(v) + "\t" + k + "\t" + " ".join(rec[k]))
    #         f.write("\n")


def viterbi_decode(emissions, transitions: torch.Tensor, start_transitions: torch.Tensor, end_transitions: torch.Tensor,
                   mask=None):  # todo
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    emissions = emissions.transpose(1, 0)
    if mask is None:
        mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list


def deocde_change(pre_list):
    """
    1.去除单个偶数字
    2. 修改类似 3, 2, 2, -> 1, 2, 2,
    todo 应该是crf的问题
    """
    res_list = pre_list[:]
    tag = []
    start_pos = -1
    for p, t in enumerate(pre_list):
        if (t == 0 or t % 2 == 1) and tag:
            if tag[0] % 2 == 0:
                res_list[start_pos:start_pos + len(tag)] = [0] * len(tag)
            elif len(tag) > 1 and all(tag[i] == tag[1] for i in range(1, len(tag))):
                res_list[start_pos] = tag[1] - 1
            start_pos = -1
            tag = []
            if t % 2 == 1:
                tag.append(t)
        elif t != 0:
            tag.append(t)
            if start_pos == -1:
                start_pos = p
    if tag and tag[0] % 2 == 0:
        res_list[start_pos:start_pos + len(tag)] = [0] * len(tag)
    elif len(tag) > 1 and all(tag[i] == tag[1] for i in range(1, len(tag))):
        res_list[start_pos] = tag[1] - 1
    return res_list


if __name__ == '__main__':
    ...
