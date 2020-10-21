# -*- coding: utf-8 -*-
import os
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]

"""数据地址"""
txt_path = "*.txt"
ann_path = "*.ann"
Train = "train"
Test = "test"
Train_Txt = "train.txt"
Test_Txt = "test.txt"
train_data_path = root_path + "/data/train/"
test_data_path = root_path + "/data/test/"
train_new_data_path = root_path + "/data/train_new/"
test_new_data_path = root_path + "/data/test_new/"
pickle_data_path = root_path + "/data/pickle/base/{}.pkl"
pickle_all_train_data_path = root_path + "/data/pickle/all_train.pkl"          # 原始数据
pickle_all_aug_train_data_path = root_path + "/data/pickle/all_aug_train.pkl"  # 增强后的数据
statist_data_path = root_path + "/data/statist/statist.txt"
statist_entity_data_path = root_path + "/data/statist/statist_entity.pkl"
submit_data_path = root_path + "/data/test"
submit_save_path = root_path + "/data/submit"

"""日志路径"""
Log = root_path + "/log/"
Result_Record = root_path + "/log/record.txt"
"""模型位置"""
pretrain_bert = root_path + "/pretrain_model/bert"
pretrain_roberta = root_path + "/pretrain_model/roberta"
pretrain_nezha = root_path + "/pretrain_model/nezha"
pretrain_txt_roberta = root_path + "/pretrain_model/pretrain"
Vocab_Path = root_path + "/pretrain_model/bert/vocab.txt"

"""模型保存"""
SAVE_MODEL_DIR = root_path + "/save_model"

"""训练参数"""
BATCH_SIZE = 4
MAX_LENGTH = 400  # 文档长度
EPOCH = 12
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
LR = 1e-5
WEIGHT_DECAY = 0.005
EMB_SIZE = 200

"""预训练参数"""
max_seq_length = 128
max_predictions_per_seq = 20
random_seed = 12345
masked_lm_prob = 0.15
dupe_factor = 10
short_seq_prob = 0.1
do_whole_word_mask = False
pretrain_batch_size = 4
train_epoch = 10
pretrain_save_path = root_path + "/pretrain_model/pretrain/"

"""其他"""
k_folds = 7
cv_pickle = root_path + "/data/pickle/{}/{}.pkl"
# seed =
