# -*- coding: utf-8 -*-
"""https://github.com/brightmart/roberta_zh/blob/master/create_pretraining_data.py修改 两个类太繁琐， 代码质量差 ，无视吧"""

from glob import glob
import collections
from transformers import BertTokenizer
import jieba
from src.config import *
import re
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

tokenization = None
common_compile = re.compile(r'[。！？?]')
jieba.load_userdict(statist_data_path)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = []

    print("create_training_instances.started...")
    for input_file in input_files:
        document = []
        content = open(input_file, encoding="utf-8").read()
        content = re.sub("\s+", "", content)
        sentences = common_compile.split(content)
        for sentence in sentences:
            document.append([i for i in sentence])
        all_documents.append(document)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    print("create_training_instances.ended...")

    return instances


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True


def get_new_segment(segment):  # 新增的方法 ####
    """
    输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    :param segment: 一句话
    :return: 一句处理过的话
    """
    seq_cws = jieba.lcut("".join(segment))
    seq_cws_dict = {x: 1 for x in seq_cws}
    new_segment = []
    i = 0
    while i < len(segment):
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 不是中文的，原文加进去。
            new_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(segment):
                continue
            if ''.join(segment[i:i + length]) in seq_cws_dict:
                new_segment.append(segment[i])
                for l in range(1, length):
                    new_segment.append('##' + segment[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i += 1
    return new_segment


def get_raw_instance(document, max_sequence_length):  # 新增的方法
    """
    获取初步的训练实例，将整段按照max_sequence_length切分成多个部分,并以多个处理好的实例的形式返回。
    :param document: 一整段
    :param max_sequence_length:
    :return: a list. each element is a sequence of text
    """
    max_sequence_length_allowed = max_sequence_length - 2
    document = [seq for seq in document if len(seq) < max_sequence_length_allowed]
    sizes = [len(seq) for seq in document]

    result_list = []
    curr_seq = []  # 当前处理的序列
    sz_idx = 0
    while sz_idx < len(sizes):
        # 当前句子加上新的句子，如果长度小于最大限制，则合并当前句子和新句子；否则即超过了最大限制，那么做为一个新的序列加到目标列表中
        if len(curr_seq) + sizes[sz_idx] <= max_sequence_length_allowed:  # or len(curr_seq)==0:
            curr_seq += document[sz_idx]
            sz_idx += 1
        else:
            result_list.append(curr_seq)
            curr_seq = []
    # 对最后一个序列进行处理，如果太短的话，丢弃掉。
    if len(curr_seq) > max_sequence_length_allowed / 2:  # /2
        result_list.append(curr_seq)

    # # 计算总共可以得到多少份
    # num_instance=int(len(big_list)/max_sequence_length_allowed)+1
    # print("num_instance:",num_instance)
    # # 切分成多份，添加到列表中
    # result_list=[]
    # for j in range(num_instance):
    #     index=j*max_sequence_length_allowed
    #     end_index=index+max_sequence_length_allowed if j!=num_instance-1 else -1
    #     result_list.append(big_list[index:end_index])
    return result_list


def create_instances_from_document(  # 新增的方法
        # 目标按照RoBERTa的思路，使用DOC-SENTENCES，并会去掉NSP任务: 从一个文档中连续的获得文本，直到达到最大长度。如果是从下一个文档中获得，那么加上一个分隔符
        #  document即一整段话，包含多个句子。每个句子叫做segment.
        # 给定一个document即一整段话，生成一些instance.
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]

    instances = []
    raw_text_list_list = get_raw_instance(document, max_seq_length)  # document即一整段话，包含多个句子。每个句子叫做segment.
    for j, raw_text_list in enumerate(raw_text_list_list):
        raw_text_list = get_new_segment(raw_text_list)  # 结合分词的中文的whole mask设置即在需要的地方加上“##”
        # 1、设置token, segment_ids
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in raw_text_list:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        # 2、调用原有的方法
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_instances_from_document_original(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    print("document_index:", document_index, "document:", type(document), " ;document:",
          document)  # document即一整段话，包含多个句子。每个句子叫做segment.
    while i < len(document):
        segment = document[i]  # 取到一个部分（可能是一段话）
        print("i:", i, " ;segment:", segment)
        ####################################################################################################################
        segment = get_new_segment(segment)  # 结合分词的中文的whole mask设置即在需要的地方加上“##”
        ###################################################################################################################
        current_chunk.append(segment)
        current_length += len(segment)
        print("#####condition:", i == len(document) - 1 or current_length >= target_seq_length)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 else \
                        tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in tqdm(enumerate(examples), desc="Converting Feature"):
        tokens = example.tokens
        segment_ids = example.segment_ids
        masked_lm_positions = example.masked_lm_positions
        masked_lm_labels = example.masked_lm_labels
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-100)
        lm_label_array[masked_lm_positions] = masked_label_ids

        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array)
        features.append(feature)
    return features


def build_dataloader(train_features):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=pretrain_batch_size)
    return train_dataloader


def main_(path):
    tokenizer = BertTokenizer.from_pretrained(Vocab_Path)
    train_path, test_path = path

    input_files = glob(train_path) + glob(test_path)

    print("*** 开始读取文件 ***")
    #
    rng = random.Random(random_seed)
    instances = create_training_instances(
        input_files, tokenizer, max_seq_length, dupe_factor,
        short_seq_prob, masked_lm_prob, max_predictions_per_seq,
        rng)

    train_features = convert_examples_to_features(instances, max_seq_length, tokenizer)
    train_dataloader = build_dataloader(train_features)
    return train_dataloader
