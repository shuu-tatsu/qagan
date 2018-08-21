#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
単語のインデックス辞書の作成
文字列をインデックス列に変換
'''

import sys
sys.path.append('./')
import codecs
import collections
import torch
import pickle
import utils
import load
import config
import torch.nn as nn
import os


class Indexer():

    def __init__(self):
        self.word2index = collections.defaultdict(int)

    def index(self,
              data_file,
              make_word_dict,
              type):
        with open(data_file, 'r') as r:
            positive_data_tuple, negative_data_tuple = load.load(file=data_file)
            # positive_data_tuple, negative_data_tuple = load_toy_data.toy_load(file=data_file)

        # 質問文と回答文を文字列で取得し，単語に分割．
        # タプル：(質問文リスト，回答文リスト)
        # 各リスト：２重構造　[[], [], ..., []]
        positive_data_tuple = qa_seq2qa_words(positive_data_tuple)
        negative_data_tuple = qa_seq2qa_words(negative_data_tuple)

        # 文字列単語とインデックス対応づけ辞書の作成
        if make_word_dict:
            sentences_questions_and_answers_list = positive_data_tuple[0] + positive_data_tuple[1]
            self.word2index = make_dictionary(self.word2index,
                                              sentences_list=sentences_questions_and_answers_list)
            with open(config.vocab_pickled_file, 'wb') as file_path:
                print('Dumping vocab_pickled_file now ...')
                pickle.dump(self.word2index, file_path)
                print('Complete dumping vocab_pickled_file')
        else:
            with open(config.vocab_pickled_file, 'rb') as file_path:
                print('Loading vocab_pickled_file now ...')
                self.word2index = pickle.load(file_path)
                print('Complete loading vocab_pickled_file')

        # 質問文と回答文のインデックス列を取得
        positive_index_data_tuple = self.get_index(positive_data_tuple)
        negative_index_data_tuple = self.get_index(negative_data_tuple)
        index_tuple_pickled = self.word2index, positive_index_data_tuple, negative_index_data_tuple

        if type == 'train':
            with open(config.train_data_pickled_file, 'wb') as file_path:
                pickle.dump(index_tuple_pickled, file_path)
        elif type == 'dev':
            with open(config.dev_data_pickled_file, 'wb') as file_path:
                pickle.dump(index_tuple_pickled, file_path)

    def index_with_input_lang(self,
              data_file,
              make_word_dict,
              type,
              input_lang):
        with open(data_file, 'r') as r:
            positive_data_tuple, negative_data_tuple = load.load_quora(file=data_file)
            # positive_data_tuple, negative_data_tuple = load_toy_data.toy_load(file=data_file)

        # 質問文と回答文を文字列で取得し，単語に分割．
        # タプル：(質問文リスト，回答文リスト)
        # 各リスト：２重構造　[[], [], ..., []]
        positive_data_tuple = qa_seq2qa_words(positive_data_tuple)
        negative_data_tuple = qa_seq2qa_words(negative_data_tuple)

        # 文字列単語とインデックス対応づけ辞書の作成
        if make_word_dict:
            sentences_questions_and_answers_list = positive_data_tuple[0] + positive_data_tuple[1]
            self.word2index = make_dictionary(self.word2index,
                                              sentences_list=sentences_questions_and_answers_list)
            with open(config.vocab_pickled_file, 'wb') as file_path:
                print('Dumping vocab_pickled_file now ...')
                pickle.dump(self.word2index, file_path)
                print('Complete dumping vocab_pickled_file')
        else:
            self.word2index = input_lang.word2index
            '''
            with open(config.vocab_pickled_file, 'rb') as file_path:
                print('Loading vocab_pickled_file now ...')
                self.word2index = pickle.load(file_path)
                print('Complete loading vocab_pickled_file')
            '''

        # 質問文と回答文のインデックス列を取得
        positive_index_data_tuple = self.get_index(positive_data_tuple)
        negative_index_data_tuple = self.get_index(negative_data_tuple)
        index_tuple_pickled = self.word2index, positive_index_data_tuple, negative_index_data_tuple

        if type == 'train':
            with open(config.train_data_pickled_file, 'wb') as file_path:
                pickle.dump(index_tuple_pickled, file_path)
        elif type == 'dev':
            with open(config.dev_data_pickled_file, 'wb') as file_path:
                pickle.dump(index_tuple_pickled, file_path)

    def transform_sents2index(self, sentences_list):
        unk_word_id = len(self.word2index) - 1
        sentences_index = [[self.word2index.get(word, unk_word_id) for word in sentence] \
                                                                   for sentence in sentences_list]
        return sentences_index

    def get_index(self, data_tuple):
        questions_list, answers_list = data_tuple
        questions_index_list = self.transform_sents2index(questions_list)
        answers_index_list = self.transform_sents2index(answers_list)
        return questions_index_list, answers_index_list


def qa_seq2qa_words(data_tuple):
    questions_list, answers_list = data_tuple
    token_questions_list = seq2words(questions_list)
    token_answers_list = seq2words(answers_list)
    return token_questions_list, token_answers_list


def seq2words(sequences_list):
    normaliszed_seq_list = [utils.normalize_string(seq) for seq in sequences_list]
    token_seq_list = [seq.split() for seq in normaliszed_seq_list]
    return token_seq_list


def make_dictionary(word2index, sentences_list, vocab_size=50000):
    # <eos>を排除
    sentences_list = [sentence[:-1]for sentence in sentences_list]
    # 2重のリストをフラットにする
    words_list = utils.flatten(sentences_list)
    # 頻度順にソートしてidをふる
    counter = collections.Counter()
    counter.update(words_list)
    cnt = 2
    for word, count in counter.most_common():
        # 出現回数３回以上の単語のみ辞書に追加
        if cnt >= vocab_size:
           break
        if count >= 3:
            word2index[word] = cnt
            cnt += 1
    word2index[u'<sos>'] = 0
    word2index[u'<eos>'] = 1
    word2index[u'<unk>'] = len(word2index)
    return word2index


def index2word(word2index, index):
    index2word = {v:k for k, v in word2index.items()}
    return index2word[index]


def index_list2word_seq(vocab, index_list, remove_eos=True):
    word_list = [index2word(vocab, word_index) for word_index in index_list]
    word_seq = ' '.join(word_list[:-1])
    return word_seq
