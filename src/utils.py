#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import re
import time
import math
import os
import pickle
import config
import index


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s) # \1: 一つ目の括弧の中身と同じ内容
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    return s


#2重のリストをフラットにする関数
def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def tensor2list(seq_tensor):
    seq_list = flatten(seq_tensor.numpy().tolist())
    return seq_list


# Pickle Loading
def load_data(data_file, type):
    if type == 'train':
        if os.path.exists(config.train_data_pickled_file):
            print('Train_data_pickled_file is already exists. Loading now ...')
        else:
            print('Train_data_pickled_file is not exists. Indexing now ...')
            indexer = index.Indexer()
            indexer.index(data_file, make_word_dict=True, type='train')
            print('Indexing is done. Loading now ...')
        with open(config.train_data_pickled_file, 'rb') as file_path:
            index_tuple_pickled = pickle.load(file_path)

    if type == 'dev':
        if os.path.exists(config.dev_data_pickled_file):
            print('Dev_data_pickled_file is already exists. Loading now ...')
        else:
            print('Dev_data_pickled_file is not exists. Indexing now ...')
            indexer = index.Indexer()
            indexer.index(data_file, make_word_dict=False, type='dev')
            print('Indexing is done. Loading now ...')
        with open(config.dev_data_pickled_file, 'rb') as file_path:
            index_tuple_pickled = pickle.load(file_path)

    return index_tuple_pickled

# Pickle Loading
def load_data_with_input_lang(data_file, type, input_lang):
    if type == 'train':
        if os.path.exists(config.train_data_pickled_file):
            print('Train_data_pickled_file is already exists. Loading now ...')
        else:
            print('Train_data_pickled_file is not exists. Indexing now ...')
            indexer = indexIndexer()
            indexer.index_with_input_lang(data_file, True, 'train', input_lang)
            print('Indexing is done. Loading now ...')
        with open(configtrain_data_pickled_file, 'rb') as file_path:
            index_tuple_pickled = pickle.load(file_path)

    if type == 'dev':
        if os.path.exists(config.dev_data_pickled_file):
            print('Dev_data_pickled_file is already exists. Loading now ...')
        else:
            print('Dev_data_pickled_file is not exists. Indexing now ...')
            indexer = index.Indexer()
            indexer.index_with_input_lang(data_file, False, 'dev', input_lang)
            print('Indexing is done. Loading now ...')
        with open(config.dev_data_pickled_file, 'rb') as file_path:
            index_tuple_pickled = pickle.load(file_path)

    return index_tuple_pickled

