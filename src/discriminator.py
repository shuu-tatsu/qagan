#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
タスク：２つのセンテンス間の関連度を算出する．
用途：QAとCDからの引用．
'''

import sys
sys.path.append('./')
import utils
import load
import index
import glove_pre_trained_vectors
from torch.nn.parameter import Parameter
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import config
import os
import pickle


class EncoderRNN(nn.Module):

    def __init__(self, vocab, hidden_size, glove_file, embedding_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.vocab = vocab
        vocab_size = vocab.n_words
        # one_hot と embeds の内積を取って，センテンス中の各単語をベクトル化し，
        # それらを concat したセンテンスベクトルを取得
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # weight の glove による初期化
        self.embedding.weight = self.glove_init(glove_file, vocab, embedding_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = self.gru(embedded, hidden)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=config.device)

    def glove_init(self, glove_file, vocab, embedding_dim):
        if os.path.exists(config.glove_pre_trained_pickled_file):
            print('Glove_pre_trained_pickled_file is already exists. Loading now ...')
        else:
            print('Glove_pre_trained_pickled_file is not exists. Getting weight now ...')
            glove_loader = glove_pre_trained_vectors.GloVeLoader(glove_file, vocab, embedding_dim)
            glove_loader.get_weight()

        with open(config.glove_pre_trained_pickled_file, 'rb') as file_path:
            weight = pickle.load(file_path)
        return Parameter(weight)


class Classifier(nn.Module):

    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, answer_tensor, question_tensor):
        '''
        回答センテンス→固定長ベクトルA
        質問センテンス→固定長ベクトルQ
        固定長ベクトルAと固定長ベクトルQのconcatをLogSoftmaxに通し，2次元テンソルで出力
        '''
        # Feed answer into GRU and get fixed answer vector
        answer_fixed_tensor = self.get_fiexed_vector(answer_tensor)

        # Feed question into GRU and get fixed question vector
        question_fixed_tensor = self.get_fiexed_vector(question_tensor)

        vectors_cat_tensor = torch.cat((question_fixed_tensor, answer_fixed_tensor), dim=1)

        # LogSoftmax() whose output dimention size is 2
        sequence_fixed_tensor = F.log_softmax(self.linear2(F.relu(self.linear(vectors_cat_tensor))), dim=1)

        return sequence_fixed_tensor

    def get_fiexed_vector(self, sequence_tensor):
        encoder_hidden = self.encoder.initHidden()
        answer_length = sequence_tensor.size(0)
        # encoder_outputs = []
        for ei in range(answer_length):
            encoder_hidden = self.encoder.forward(sequence_tensor[ei], encoder_hidden)
            # encoder_outputs.append(encoder_output)
        # return torch.mean(torch.cat(encoder_outputs, dim=0), dim=0, keepdim=True)
        return encoder_hidden


def tensor_from_sentence(sentence):
    return torch.tensor(sentence, dtype=torch.long, device=config.device).view(-1, 1)


def tensors_from_pair(pair):
    # answerを入力，questionを出力
    # pair[1]: answer_index_list as input
    # pair[0]: question_index_list as target
    # pair[2]: label
    input_tensor = tensor_from_sentence(pair[1])
    target_tensor = tensor_from_sentence(pair[0])
    return (input_tensor, target_tensor, pair[2])


def random_choice_pair_from_pairs(pairs):
    num_pairs = len(pairs[0])
    i = random.randrange(num_pairs)
    return (pairs[0][i], pairs[1][i], pairs[2][i])


def get_label_tensor(label_int):
    if label_int == 1: # positive
        return torch.LongTensor([1])
    elif label_int == 0: # negative
        return torch.LongTensor([0])
    else:
        print('Labeling Error')


def train_iters(train_pairs, dev_pairs, model, n_iters, batch_size,
                print_every, learning_rate=config.args.learning_rate):
    # pairs[0]: questions_index_list
    # pairs[1]: answers_index_list
    # pairs[2]: label 0 or 1
    #
    # pairs: ([[q1_index_list], [q2_index_list], ..., [qn_index_list]],
    #         [[a1_index_list], [a2_index_list], ..., [an_index_list]],
    #         [[l1_int], [l2_int], ..., [ln_int]])
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    # optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_pairs_list = [tensors_from_pair(random_choice_pair_from_pairs(train_pairs))
                           for i in range(n_iters)]
    criterion = nn.NLLLoss()
    batch_loss = 0.0

    # 1 iter につき，1つのQAペア
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs_list[iter - 1]
        answer_tensor = training_pair[0]
        question_tensor = training_pair[1]
        label_int = training_pair[2]

        '''
        answer_tensor:
        tensor([[ 6],
                [ 8],
                [ 7],
                [ 2],
                [ 1]])

        question_tensor:
        tensor([[    56],
                [ 26621],
                [     5],
                [  6440],
                [  4177],
                [  1797],
                [     1]])

        label_int:
        1
        '''
        # loss: NLLLoss
        y = model.forward(answer_tensor, question_tensor)
        label_tensor = get_label_tensor(label_int).to(config.device)
        loss = criterion(y, label_tensor)

        print_loss_total += loss.item()

        batch_loss += loss
        if iter % batch_size == 0:
            # print(model.encoder.gru.)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0.0

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # 訓練データでの評価
            train_score = evaluate_randomly(train_pairs, model, n_iters=100)
            # 開発データでの評価
            dev_score = evaluate_randomly(dev_pairs, model, n_iters=100)
            print('Time:%s (%d %d%%) Loss:%.4f Accuracy(train data):%s Accuracy(dev data):%s' % (utils.time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, train_score, dev_score))


def evaluate_randomly(pairs, model, n_iters):
    count = 0
    for i in range(n_iters):
        # pairsの中からランダムに1組のpairを取得し，tensor型に変換
        training_pair_list = tensors_from_pair(random_choice_pair_from_pairs(pairs))
        answer_tensor = training_pair_list[0]
        question_tensor = training_pair_list[1]
        label_int = training_pair_list[2]
        with torch.no_grad():
            predicted_label_tensor = model.forward(answer_tensor, question_tensor)
        negative_probability = predicted_label_tensor[0][0].item()
        positive_probability = predicted_label_tensor[0][1].item()
        if positive_probability >= negative_probability:
            predicted_label = 1
        else:
            predicted_label = 0

        if predicted_label == label_int:
            count += 1
    accuracy = count / n_iters
    return accuracy


def labeling(sequences_index_tuple, label):
    sequences_size = len(sequences_index_tuple[0])
    sequences_label_list = [label for i in range(sequences_size)]
    sequences_index_tuple_labeled = (sequences_index_tuple[0],
                                     sequences_index_tuple[1],
                                     sequences_label_list)
    return sequences_index_tuple_labeled


def merge_posi_nega_data(positive_tuple, negative_tuple):
    questions = positive_tuple[0] + negative_tuple[0]
    answers = positive_tuple[1] + negative_tuple[1]
    labels = positive_tuple[2] + negative_tuple[2]
    pairs = (questions, answers, labels)
    return pairs


def get_pairs(data_file, type):
    vocab, positive_index_tuple, negative_index_tuple = utils.load_data(data_file, type)
    positive_index_tuple_labeled = labeling(positive_index_tuple, label=1)
    negative_index_tuple_labeled = labeling(negative_index_tuple, label=0)
    pairs = merge_posi_nega_data(positive_index_tuple_labeled, negative_index_tuple_labeled)
    return vocab, pairs

def get_positive_pairs(data_file, type, input_lang):
    vocab, positive_index_tuple, negative_index_tuple = utils.load_data_with_input_lang(data_file, type, input_lang)
    positive_index_tuple_labeled = labeling(positive_index_tuple, label=1)
    pairs = positive_index_tuple_labeled
    return vocab, pairs

def main():
    torch.manual_seed(1)

    # loading training data as train_pairs
    # pairs の前半分はポジティブデータ、後半分はネガティブデータ
    vocab, train_pairs = get_pairs(data_file=config.quora_train_file, type='train')
    # loading eval data as eval_pairs
    _, dev_pairs = get_pairs(data_file=config.quora_dev_file, type='dev')
    # pairs[0]: questions_index_list
    # pairs[1]: answers_index_list
    # pairs[2]: label 0 or 1
    #
    # pairs: ([[q1_index_list], [q2_index_list], ..., [qn_index_list]],
    #         [[a1_index_list], [a2_index_list], ..., [an_index_list]],
    #         [[l1_int], [l2_int], ..., [ln_int]])
    print('QUORA    Train size:{}    Dev size:{}    Vocab size:{}'.format(len(train_pairs[0]), len(dev_pairs[0]), len(vocab)))
    use_toy = True
    if use_toy:
        train_size = 5000
        dev_size = 500
        train_pairs = ((train_pairs[0][:train_size // 2] + train_pairs[0][-train_size // 2:]),
                       (train_pairs[1][:train_size // 2] + train_pairs[1][-train_size // 2:]),
                       (train_pairs[2][:train_size // 2] + train_pairs[2][-train_size // 2:]))
        dev_pairs = ((dev_pairs[0][:dev_size // 2] + dev_pairs[0][-dev_size // 2:]),
                     (dev_pairs[1][:dev_size // 2] + dev_pairs[1][-dev_size // 2:]),
                     (dev_pairs[2][:dev_size // 2] + dev_pairs[2][-dev_size // 2:]))

    else:
        train_size = len(train_pairs[0])
    hidden_size = config.args.embedding_dim
    encoder = EncoderRNN(vocab,
                         hidden_size,
                         glove_file=config.glove_file,
                         embedding_dim=config.args.embedding_dim).to(config.device)
    model = Classifier(encoder, encoder.hidden_size).to(config.device)
    print('Using    Train size:{}    Dev size:{}    Vocab size:{}'.format(len(train_pairs[0]), len(dev_pairs[0]), len(vocab)))
    train_iters(train_pairs, dev_pairs, model, n_iters=10 * train_size, batch_size=32, print_every=32)


if __name__ == '__main__':
    main()
