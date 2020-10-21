# -*- coding: utf-8 -*-
import math
from src.config import *
from src.model import *
from torchcrf import CRF

def weight_f1(t, lamb=0.25):
    '''
    # 牛顿冷却定律
    '''
    return math.exp(-lamb * t)


models = ['/bilstm_0.703', '/bilstm_0.699', '/tianchi_ner_0.696', '/tianchi_ner_0.694']
model_corr = {'bilstm': BiLstm_Ner, 'idcnn': IDCNN_NER, 'base': BaseNer}


def model_cv(model_list, inputs, mask):
    cv_output = torch.zeros(inputs.size())  # todo
    wf1 = [weight_f1(t) for t in range(len(model_list))]
    wf1_sum = sum(wf1)
    for model, wf in zip(model_list, wf1):
        _model = model.split("_")[0][1:]
        _model = model_corr[_model](pretrain_txt_roberta, EMB_SIZE)
        model_dir = SAVE_MODEL_DIR + model
        _model.load_state_dict(model_dir)
        output = _model.get_output(inputs, mask)
        weight_ouput = wf * output
        cv_output += weight_ouput
    pre_list  = 1