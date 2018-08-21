#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
gloveの事前学習時済みパラメータによるモデルパラメータの初期化．
具体的には，QGにおけるEncoderRNNのself.embedding.weightを初期化する．
'''

import sys
sys.path.append('./')
import codecs
import collections
import torch
import pickle
import utils
import load
import torch.nn as nn
import index
import os
import config


class GloVeLoader():

    def __init__(self,
                 glove_file,
                 vocab,
                 embedding_dim):
        self.glove_file = glove_file
        self.vocab = vocab
        self.embedding_dim = embedding_dim

    def read_glove(self):
        with codecs.open(self.glove_file, 'r', 'utf-8') as r:
            words = r.readlines()
        return words

    def get_train_vocaburary_set(self):
        train_vocabulary_set = set(self.vocab.word2index.keys())
        return train_vocabulary_set

    def get_train_vocaburary_size(self):
        vocab_size = self.vocab.n_words
        return vocab_size

    def get_glove_vocaburary_set(self):
        words = self.read_glove()
        glove_vocabulary = []
        for word in words:
            tok = word.split()
            glove_vocabulary.append(tok[0])
        glove_vocabulary_set = set(glove_vocabulary)
        return glove_vocabulary_set

    def extract_train_and_glove(self,
                                train_vocabulary_set,
                                glove_vocabulary_set):
        return train_vocabulary_set & glove_vocabulary_set

    def get_glove_words_vectors_dict(self):
        words = self.read_glove()
        glove_dict = {}
        for word in words:
            tok = word.split()
            glove_dict[tok[0]] = tok[1:]
        return glove_dict

    def get_common_words_vectors_dict(self, common_words, glove_words_vectors_dict):
        common_words_dict = {}
        for word, vector in glove_words_vectors_dict.items():
            if word in common_words:
                common_words_dict[word] = vector
        return common_words_dict

    def get_weight(self):
        # train_vocabulary_set と glove_vocabulary_set を取得
        train_vocab_size = self.get_train_vocaburary_size()
        train_vocabulary_set = self.get_train_vocaburary_set()
        glove_vocabulary_set = self.get_glove_vocaburary_set()

        # ランダム初期化済み weight を取得
        self.word_embeddings = nn.Embedding(train_vocab_size, self.embedding_dim)
        self.weight = self.word_embeddings.weight

        # train と glove の積集合を取得
        common_words = self.extract_train_and_glove(train_vocabulary_set,
                                                        glove_vocabulary_set)
        # glove の単語とベクトル辞書を取得
        glove_words_vectors_dict = self.get_glove_words_vectors_dict()

        # glove と train の共通単語を辞書の key とし，
        # その単語の glove におけるベクトルを value とした辞書
        common_words_dict = self.get_common_words_vectors_dict(common_words, glove_words_vectors_dict)

        for word, index in self.vocab.word2index.items():
            if word in common_words:
                # train に出てくる word が common_words に含まれていれば，
                # それの (glove 由来の) vector を取得
                word_vector = [float(vec_element) for vec_element in common_words_dict[word]]

                # 取得した vector で weight における，
                # 該当する index 上の vector を上書き
                self.weight[index] = torch.Tensor(word_vector)

        with open(config.glove_pre_trained_pickled_file, 'wb') as file_path:
            pickle.dump(self.word_embeddings.weight, file_path)
