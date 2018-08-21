#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch


parser = argparse.ArgumentParser()

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File
target_dir = '../data/'
train_file = target_dir + '/msmarco/train_v2.1.json'
dev_file = target_dir + '/msmarco/dev_v2.1.json'
eval_file = target_dir + '/msmarco/eval_v2.1_public.json'

quora_train_file = target_dir + '/quora/train.tsv'
quora_dev_file = target_dir + '/quora/dev.tsv'

train_data_pickled_file = '../data/pickled/train_data_pickled.pkl'
dev_data_pickled_file = '../data/pickled/dev_data_pickled.pkl'
vocab_pickled_file = '../data/pickled/vocab_pickled.pkl'
glove_pre_trained_pickled_file = '../data/pickled/glove_pre_trained_pickled.pkl'

# Data
parser.add_argument("--max_length", type=int, default=50, help="max_length")
parser.add_argument("--sos_token", type=int, default=0, help="sos_token")
parser.add_argument("--eos_token", type=int, default=1, help="eos_token")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning_rate")

# Dimention
#parser.add_argument("--embedding_dim", type=int, default=50, help="embedding_dim")
parser.add_argument("--embedding_dim", type=int, default=100, help="embedding_dim")
#parser.add_argument("--embedding_dim", type=int, default=200, help="embedding_dim")
#parser.add_argument("--embedding_dim", type=int, default=300, help="embedding_dim")
#glove_file = '../data/embedding/glove.6B.50d.txt'
glove_file = '../data/embedding/glove.6B.100d.txt'
#glove_file = '../data/embedding/glove.6B.200d.txt'
#glove_file = '../data/embedding/glove.6B.300d.txt'

args = parser.parse_args()
