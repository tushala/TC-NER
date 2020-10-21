# -*- coding: utf-8 -*-

""" 数据增强 自己写的 O替换"""  # todo 实体交换？
import pickle
import random
from transformers import BertTokenizer
from src.config import statist_entity_data_path, pickle_all_train_data_path, pickle_all_aug_train_data_path
from src.data_proc import Text_Example


def data_augment(tokenizer: BertTokenizer, augument_size=3):
    assert Text_Example
    aug_data = []

    entity_data = pickle.load(open(statist_entity_data_path, "rb"))
    all_train_data = pickle.load(open(pickle_all_train_data_path, "rb"))
    for data in all_train_data:
        text = data.text
        anns = data.anns
        if len(anns) < 3:
            aug_data.append(data)
            continue
        new_text = text
        for aug in range(augument_size):
            sample_anns_idx = random.sample(range(len(anns)), 2)
            for idx in sample_anns_idx:
                _tag, ann = anns[idx]
                ann_replace_list = entity_data[_tag][len(ann)]
                if len(ann_replace_list) < 2:
                    break
                repalce_ann = random.choice(ann_replace_list)
                while repalce_ann == ann:
                    repalce_ann = random.choice(ann_replace_list)
                new_text = text.replace(ann, repalce_ann, 1)

        new_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in new_text]
        new_text_example = data
        new_text_example.text = new_text
        new_text_example.token_ids = new_token_ids
        #todo  ann
# def data_augment(augument_size=3):
#     """替换O"""
#     assert Text_Example
#     aug_data = []
#
#     all_train_data = pickle.load(open(pickle_all_train_data_path, "rb"))
#     for data in all_train_data:
#         aug_data.append(data)
#         token_ids = data.tag_ids
#         otoken_ids = [i for i, t in enumerate(token_ids) if t == 0]
#         if len(otoken_ids) < 3:
#             continue
#         for i in range(augument_size):
#             new_ag_data = data
#             # new_text = new_ag_data.text
#             new_token_ids = new_ag_data.token_ids
#
#             idxs = random.sample(otoken_ids, 3)
#             new_token_ids[idxs[0]], new_token_ids[idxs[1]], new_token_ids[idxs[2]] = \
#                 new_token_ids[idxs[1]], new_token_ids[idxs[2]], new_token_ids[idxs[0]]
#
#             aug_data.append(new_ag_data)
#     random.shuffle(aug_data)
#     pickle.dump(aug_data, open(pickle_all_aug_train_data_path, "wb"))
if __name__ == '__main__':
    # all_train_data = pickle.load(open(pickle_all_train_data_path, "rb"))
    # all_aug_train_data = pickle.load(open(pickle_all_aug_train_data_path, "rb"))
    # print(len(all_train_data))
    # print(len(all_aug_train_data))
    # print(all_train_data[0])
    data_augment()