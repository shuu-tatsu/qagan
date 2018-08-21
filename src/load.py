#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
MS MARCOから質問文と回答文のペアを文字列として読み取る．
関数　get_positive_data：ポジティブなペアを文字列で取得
関数　get_negative_data：ネガティブなペアを文字列で取得
'''

import sys
sys.path.append('./')
import json
import random
import config
import csv


def get_positive_data(dev_data_dict):
    positive_questions_list = []
    positive_answers_list = []
    for query_str, answers_list, wellFormedAnswers_list in zip(dev_data_dict['query'].values(),
                                                               dev_data_dict['answers'].values(),
                                                               dev_data_dict['wellFormedAnswers'].values()):


        if len(wellFormedAnswers_list[0]) > 1:
            if len(wellFormedAnswers_list[0]) <=50 and len(query_str) <= 50:
                positive_answers_list.append(wellFormedAnswers_list[0] + ' <eos>')
                positive_questions_list.append(query_str + ' <eos>')
        else:
            if answers_list[0] == 'No Answer Present.':
                # データに加えない
                pass
            else:
                if len(answers_list[0]) <= 50 and len(query_str) <= 50:
                    positive_answers_list.append(answers_list[0] + ' <eos>')
                    positive_questions_list.append(query_str + ' <eos>')

    return positive_questions_list, positive_answers_list


def get_negative_data(data_dict):
    positive_questions_list, positive_answers_list = get_positive_data(data_dict)
    # positive_questions_list を１インデックスだけスライドさせる
    negative_questions_list = [positive_questions_list[i + 1] for i in range(len(positive_questions_list) - 1)]
    negative_questions_list.append(positive_questions_list[0])
    # answersはそのままの順序で利用
    negative_answers_list = positive_answers_list
    return negative_questions_list, negative_answers_list


def load(file):
    with open(file, 'r') as r:
        '''
        dict_keys(['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'])
        '''
        data_dict = json.load(r)
    positive_data_tuple = get_positive_data(data_dict)
    negative_data_tuple = get_negative_data(data_dict)
    return positive_data_tuple, negative_data_tuple


def get_quora(file):
    # id	qid1	qid2	question1	question2	is_duplicate
    positive_questions_list = []
    positive_answers_list = []
    negative_questions_list = []
    negative_answers_list = []
    #with open('dev.tsv') as f:
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader: 
            if len(row[3]) <= 50 and len(row[4]) <= 50:
                if row[5] == '1':
                    positive_answers_list.append(row[3] + ' <eos>')
                    positive_questions_list.append(row[4] + ' <eos>')
                elif row[5] == '0':
                    negative_answers_list.append(row[3] + ' <eos>')
                    negative_questions_list.append(row[4] + ' <eos>')
    return positive_questions_list, positive_answers_list, negative_questions_list, negative_answers_list


def load_quora(file):
    positive_questions_list, positive_answers_list, negative_questions_list, negative_answers_list = get_quora(file)
    positive_data_tuple = positive_questions_list, positive_answers_list
    negative_data_tuple = negative_questions_list, negative_answers_list
    return positive_data_tuple, negative_data_tuple


def main():
    positive_data_tuple, negative_data_tuple = load(file=config.dev_file)


if __name__ == '__main__':
    main()
