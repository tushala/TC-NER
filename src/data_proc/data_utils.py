# -*- coding: utf-8 -*-
# 重写数据获取 用于多折cv
from src.mylib import *
from glob import glob
from transformers import BertTokenizer
from collections import namedtuple
import pickle

Text_Example = namedtuple("Text_Example", ['text', 'tag', 'token_ids', 'tag_ids', 'anns'])


def get_txt_info(tokenizer: BertTokenizer, txt_content, ann_content, max_length):
    text_examples = []
    if len(txt_content) > max_length:
        cur_sententce = []
        sentences = txt_content.split("。")
        sentences = [t + '。' for t in sentences]  # 菜鸡代码
        sentences[-1] = sentences[-1][:-1]
        cur_txt = ""
        cur_idx = 0
        for i in range(len(sentences) - 1):
            if len(cur_txt) + len(sentences[i]) > max_length:
                cur_sententce.append(cur_txt)
                cur_txt = ""
                cur_idx = i
            cur_txt += sentences[i]
        if len(cur_txt) + len(sentences[-1]) < max_length:
            cur_sententce.append("".join(sentences[cur_idx:]))
        else:
            cur_sententce.append("".join(sentences[cur_idx:-1]))
            cur_sententce.append(sentences[-1])
    else:
        cur_sententce = [txt_content]
    if not all(len(i) <= max_length for i in cur_sententce):
        print(cur_sententce)
    assert all(len(i) <= max_length for i in cur_sententce)
    assert sum(len(i) for i in cur_sententce) == len(txt_content)

    def _cut(cur_sententce, ann_content):
        record = 0
        nonlocal text_examples
        cur_left = 0
        ann_left = 0
        idx = 0
        al = len(ann_content)
        for cs in cur_sententce:
            cur_right = cur_left + len(cs)
            tags = [O] * len(cs)
            for i in range(ann_left, al):
                idx = i
                if ann_content[i][0] >= cur_right:
                    idx -= 1
                    break
            for i in range(ann_left, idx + 1):
                record += 1
                start, end, tag, _ = ann_content[i]
                start -= cur_left
                end -= cur_left
                _tags = [f"B-{tag}"] + [f"I-{tag}"] * (end - start - 1)
                tags[start:end] = _tags
            anns = [(i[-2], i[-1]) for i in ann_content[ann_left:idx + 1]]
            assert len(tags) == len(cs)
            cur_left = cur_right
            ann_left = idx + 1
            token_ids = [tokenizer.convert_tokens_to_ids(i) for i in cs]
            tag_ids = [tag2indx[i] for i in tags]
            te = Text_Example(cs, tags, token_ids, tag_ids, anns)
            text_examples.append(te)
        assert record == len(ann_content)

    _cut(cur_sententce, ann_content)
    return text_examples


def get_all_data(path, tokenizer, max_length):
    text_examples = []
    entities = set()
    for txt in glob(path):
        # txt = r"E:\tinachi-ner\data\train\0.txt"
        txt_content = open(txt, encoding="utf-8").read()
        ann_path = txt.replace("txt", "ann")
        ann_content = []
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                _, sy_st_en, name = line.split("\t")
                entities.add(name)
                sy, st, en = sy_st_en.split(" ")
                ann_content.append((int(st), int(en), sy, name))
            ann_content = sorted(ann_content, key=lambda x: x[1])

        text_example = get_txt_info(tokenizer, txt_content, ann_content, max_length)
        text_examples.extend(text_example)
    with open(pickle_all_train_data_path, 'wb') as f:
        pickle.dump(text_examples, f)
    if not os.path.exists(statist_data_path):
        with open(statist_data_path, "w", encoding="utf-8") as f:
            for et in entities:
                f.write(et + " " + str(1000) + "\n")
    return text_examples


if __name__ == '__main__':
    t = BertTokenizer.from_pretrained(Vocab_Path)
    get_all_data(train_data_path + txt_path, t, MAX_LENGTH)
