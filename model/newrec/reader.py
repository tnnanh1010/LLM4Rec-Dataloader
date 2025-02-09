import os
import csv
import json
import random
import logging
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import PreTrainedTokenizer
import numpy as np
from typing import List, Tuple
from entities import News, Dataset

class Reader:
    def __init__(self, args, tokenizer: PreTrainedTokenizer = None, max_title_length: int = None, max_sapo_length: int = None, user2id: dict = None, category2id: dict = None, max_his_click: int = None, npratio: int = None):
        self.args = args
        self.nGPU = args.nGPU
        self.npratio = args.npratio
        random.seed(args.seed)
        self._tokenizer = tokenizer
        self._max_title_length = max_title_length
        self._max_sapo_length = max_sapo_length
        self._user2id = user2id
        self._category2id = category2id
        self._max_his_click = max_his_click
        self._npratio = npratio

    def read_custom_abstract(self, news_path, custom_abstract_dict):
        news = {}
        news_index = {}
        category_dict = {}
        subcategory_dict = {}
        word_cnt = {}

        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                splited = line.strip('\n').split('\t')
                doc_id, category, subcategory, title, abstract, url, entity_title, entity_abstract = splited
                if doc_id in custom_abstract_dict:
                    abstract = custom_abstract_dict[doc_id]
                news[doc_id] = [title.split(' '), category, subcategory, abstract.split(' ')]
                news_index[doc_id] = len(news_index) + 1
                for word in title.split(' '):
                    if word not in word_cnt:
                        word_cnt[word] = 0
                    word_cnt[word] += 1
                for word in abstract.split(' '):
                    if word not in word_cnt:
                        word_cnt[word] = 0
                    word_cnt[word] += 1
                if category not in category_dict:
                    category_dict[category] = len(category_dict) + 1
                if subcategory not in subcategory_dict:
                    subcategory_dict[subcategory] = len(subcategory_dict) + 1

        return news, news_index, category_dict, subcategory_dict, word_cnt

    def read_news(self, news_path, abstract_path=None, mode='train'):
        news = {}
        category_dict = {}
        subcategory_dict = {}
        news_index = {}
        word_cnt = Counter()

        if self.args.use_custom_abstract and abstract_path:
            with open(abstract_path, 'r') as f:
                abs = json.load(f)

        with open(news_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                splited = line.strip('\n').split('\t')
                doc_id, category, subcategory, title, abstract, url, _, _ = splited
                self.update_dict(news_index, doc_id)

                title = title.lower()
                title = word_tokenize(title, language='english', preserve_line=True)

                if self.args.use_custom_abstract and abstract_path:
                    abstract = abs[doc_id] if doc_id in abs else abstract

                self.update_dict(news, doc_id, [title, category, subcategory, abstract])
                if mode == 'train':
                    if self.args.use_category:
                        self.update_dict(category_dict, category)
                    if self.args.use_subcategory:
                        self.update_dict(subcategory_dict, subcategory)
                    word_cnt.update(title)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > self.args.filter_num]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            return news, news_index, category_dict, subcategory_dict, word_dict
        elif mode == 'test':
            return news, news_index
        else:
            assert False, 'Wrong mode!'

    def update_dict(self, dictionary, key, value=None):
        if key not in dictionary:
            if value is None:
                dictionary[key] = len(dictionary) + 1
            else:
                dictionary[key] = value
    
    def get_doc_input(self, news, news_index, category_dict, subcategory_dict, word_dict, args):
        news_num = len(news) + 1
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
        news_subcategory = np.zeros((news_num, 1), dtype='int32') if args.use_subcategory else None
        news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32') if args.use_abstract else None

        for key in tqdm(news):
            title, category, subcategory, abstract = news[key]
            doc_index = news_index[key]

            for word_id in range(min(args.num_words_title, len(title))):
                if title[word_id] in word_dict:
                    news_title[doc_index, word_id] = word_dict[title[word_id]]

            if args.use_category:
                news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
            if args.use_subcategory:
                news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
            if args.use_abstract:
                for word_id in range(min(args.num_words_abstract, len(abstract))):
                    if abstract[word_id] in word_dict:
                        news_abstract[doc_index, word_id] = word_dict[abstract[word_id]]

        return news_title, news_category, news_subcategory, news_abstract

    def get_sample(self, all_elements, num_sample):
        if num_sample > len(all_elements):
            return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
        else:
            return random.sample(all_elements, num_sample)

    def prepare_training_data(self, train_data_dir):
        behaviors = []

        behavior_file_path = os.path.join(train_data_dir, 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')
                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = self.get_sample(neg, self.npratio)
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    behaviors.append(new_line)

        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(self.nGPU)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % self.nGPU].append(line)

        logging.info('Writing files...')
        for i in range(self.nGPU):
            processed_file_path = os.path.join(train_data_dir, f'behaviors_np{self.npratio}_{i}.tsv')
            with open(processed_file_path, 'w') as f:
                f.writelines(behaviors_per_file[i])

        return len(behaviors)

    def prepare_testing_data(self, test_data_dir):
        behaviors = [[] for _ in range(self.nGPU)]

        behavior_file_path = os.path.join(test_data_dir, 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors[i % self.nGPU].append(line)

        logging.info('Writing files...')
        for i in range(self.nGPU):
            processed_file_path = os.path.join(test_data_dir, f'behaviors_{i}.tsv')
            with open(processed_file_path, 'w') as f:
                f.writelines(behaviors[i])

        return sum([len(x) for x in behaviors])

    def load_matrix(self, embedding_file_path, word_dict, word_embedding_dim):
        embedding_matrix = np.zeros(shape=(len(word_dict) + 1, word_embedding_dim))
        have_word = []
        if embedding_file_path is not None:
            with open(embedding_file_path, 'rb') as f:
                while True:
                    line = f.readline()
                    if len(line) == 0:
                        break
                    line = line.split()
                    word = line[0].decode()
                    if word in word_dict:
                        index = word_dict[word]
                        tp = [float(x) for x in line[1:]]
                        embedding_matrix[index] = np.array(tp)
                        have_word.append(word)
        return embedding_matrix, have_word

    def read_train_dataset(self, data_name: str, news_path: str, behaviors_path: str):
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_train_line(i, line, news_dataset, dataset)

        return dataset

    def read_eval_dataset(self, data_name: str, news_path: str, behaviors_path: str) :
        dataset, news_dataset = self._read(data_name, news_path)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                self._parse_eval_line(i, line, news_dataset, dataset)

        return dataset

    def _read(self, data_name: str, news_path: str):
        dataset = Dataset(data_name, self._tokenizer, self._category2id)
        news_dataset = self._read_news_info(news_path, dataset)

        return dataset, news_dataset

    def _read_news_info(self, news_path: str, dataset: Dataset) -> dict:
        pad_news_obj = dataset.create_news(
            [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id],
            [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id], self._category2id['pad'])
        news_dataset = {'pad': pad_news_obj}
        with open(news_path, mode='r', encoding='utf-8', newline='') as f:
            news_tsv = csv.reader(f, delimiter='\t')
            for line in news_tsv:
                title_encoding = self._tokenizer.encode(line[1], add_special_tokens=True, truncation=True,
                                                        max_length=self._max_title_length)
                category_id = self._category2id.get(line[2], self._category2id['unk'])
                sapo_encoding = self._tokenizer.encode(line[3], add_special_tokens=True, truncation=True,
                                                       max_length=self._max_sapo_length)
                news = dataset.create_news(title_encoding, sapo_encoding, category_id)
                news_dataset[line[0]] = news

        return news_dataset

    def _parse_train_line(self, impression_id, line, news_dataset, dataset):
        user_id = self._user2id.get(line[1], self._user2id['unk'])
        history_clicked = [news_dataset[news_id] for news_id in line[3].split()]
        history_clicked = [news_dataset['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[:self._max_his_click]
        pos_news = [news_dataset[news_id] for news_id, label in [behavior.split('-') for behavior in line[4].split()] if label == '1']
        neg_news = [news_dataset[news_id] for news_id, label in [behavior.split('-') for behavior in line[4].split()] if label == '0']
        for news in pos_news:
            label = [1] + [0] * self._npratio
            list_news = [news] + sample_news(neg_news, self._npratio, news_dataset['pad'])
            impression_news = list(zip(list_news, label))
            random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            impression = dataset.create_impression(impression_id, user_id, list_news, label)
            dataset.add_sample(user_id, history_clicked, impression)

    def _parse_eval_line(self, impression_id, line, news_dataset, dataset):
        user_id = self._user2id.get(line[1], self._user2id['unk'])
        history_clicked = [news_dataset[news_id] for news_id in line[3].split()]
        history_clicked = [news_dataset['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[:self._max_his_click]
        for behavior in line[4].split():
            news_id, label = behavior.split('-')
            impression = dataset.create_impression(impression_id, user_id, [news_dataset[news_id]], [int(label)])
            dataset.add_sample(user_id, history_clicked, impression)


def sample_news(list_news: List[News], num_news: int, pad: News) -> List:
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))