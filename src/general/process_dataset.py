import csv
import os
import random
import re
import xml.etree.ElementTree

import numpy as np
from sklearn.model_selection import train_test_split

from general.helpers_text import get_splits, purge_authors

random.seed(42)


# --------
# general methods for data processing
# --------

# divide dataset into training, validation and test sets
def divide_dataset(texts, labels, only_train_val=False):
    x_trval, x_te, y_trval, y_te = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)
    if only_train_val:
        return x_trval, x_te, y_trval, y_te
    else:
        x_tr, x_val, y_tr, y_val = train_test_split(x_trval, y_trval, test_size=0.1, random_state=42, stratify=y_trval)
        return x_tr, x_val, x_te, y_tr, y_val, y_te


# --------
# processing methods for each specific dataset
# --------

# victoria dataset of '800 English novels
def process_victoria(data_path='../dataset/', **kwargs):
    data_path = data_path + 'victoria/Gungor_2018_VictorianAuthorAttribution_data-train.csv'
    print('Creating dataset Victoria...')
    texts = []
    labels = []
    with open(data_path, 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            splits = get_splits(row[0])
            texts.extend(splits)
            labels.extend([row[1]] * len(splits))
    authors = np.unique(labels)
    selected_indices = []
    for author in authors:
        indices = np.where(np.array(labels) == author)[0].tolist()
        if len(indices) > 1000:
            selected_indices.extend(random.sample(indices, 1000))
        else:
            selected_indices.extend(indices)
    selected_texts = [texts[index] for index in selected_indices]
    selected_labels = [labels[index] for index in selected_indices]
    x_tr, x_val, x_te, y_tr, y_val, y_te = divide_dataset(selected_texts, selected_labels, only_train_val=False)
    #x_tr, x_val, x_te, y_tr, y_val, y_te = divide_dataset(texts, labels, only_train_val=False)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr)
    selected_authors = random.sample(authors.tolist(), 5)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr, to_keep=selected_authors)
    x_val, y_val, _ = purge_authors(x_val, y_val, to_keep=selected_authors)
    x_te, y_te, _ = purge_authors(x_te, y_te, to_keep=selected_authors)
    return x_tr, x_val, x_te, y_tr, y_val, y_te


# TweepFake dataset (Twitter)
# dataset with 17 human accounts and 23 bots (imitating one human account)
def process_tweepfake(data_path='../dataset/', **kwargs):
    data_path = data_path + 'TweepFake/'
    print('Creating dataset TweepFake...')
    x_tr, x_val, x_te = [], [], []
    y_tr, y_val, y_te = [], [], []
    with open(data_path + 'full_train.csv', 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)
        for user_id, status_id, screen_name, text, account_type, class_type, screen_name_anonymized in csv_reader:
            if account_type == 'human':
                splits = get_splits(text)
                x_tr.extend(splits)
                y_tr.extend([screen_name_anonymized] * len(splits))
    x_tr, y_tr, to_keep = purge_authors(x_tr, y_tr)
    with open(data_path + 'full_validation.csv', 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)
        for user_id, status_id, screen_name, text, account_type, class_type, screen_name_anonymized in csv_reader:
            if account_type == 'human':
                splits = get_splits(text)
                x_val.extend(splits)
                y_val.extend([screen_name_anonymized] * len(splits))
    x_val, y_val, _ = purge_authors(x_val, y_val, to_keep)
    with open(data_path + 'full_test.csv', 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)
        for user_id, status_id, screen_name, text, account_type, class_type, screen_name_anonymized in csv_reader:
            splits = get_splits(text)
            x_te.extend(splits)
            y_te.extend([screen_name_anonymized] * len(splits))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


# EBG-obfuscation
def process_ebg(data_path='../dataset/', **kwargs):
    data_path = data_path + 'EBG-obfuscation/'
    texts, labels = [], []
    for file in os.listdir(data_path + 'train'):
        with open(data_path + 'train/' + file, 'r') as data_file:
            splits = get_splits(data_file.read())
            texts.extend(splits)
            labels.extend([file.split('_', maxsplit=1)[0]] * len(splits))
    if kwargs['obfuscation']:
        # dividing train-val, taking the obfuscated test set
        print('Creating dataset EBG-obfuscation...')
        x_tr, x_val, y_tr, y_val = divide_dataset(texts, labels, only_train_val=True)
        x_te, y_te = [], []
    else:
        # dividing into train-val-test
        print('Creating dataset EBG...')
        x_tr, x_val, x_te, y_tr, y_val, y_te = divide_dataset(texts, labels, only_train_val=False)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr)
    selected_authors = random.sample(np.unique(y_tr).tolist(), 10)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr, to_keep=selected_authors)
    x_val, y_val, _ = purge_authors(x_val, y_val, to_keep=selected_authors)
    if kwargs['obfuscation']:
        for file in os.listdir(data_path + 'test'):
            with open(data_path + 'test/' + file, 'r') as data_file:
                splits = get_splits(data_file.read())
                x_te.extend(splits)
                y_te.extend([file.split('_', 1)[0]] * len(splits))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


# RJ-obfuscation
def process_rj(data_path='../dataset/', **kwargs):
    data_path = data_path + 'RJ-obfuscation/'
    texts, labels = [], []
    for file in os.listdir(data_path + 'train'):
        with open(data_path + 'train/' + file, 'r') as data_file:
            splits = get_splits(data_file.read())
            texts.extend(splits)
            labels.extend([file.split('_', maxsplit=1)[0]] * len(splits))
    if kwargs['obfuscation']:
        # dividing train-val, taking the obfuscated test set
        print('Creating dataset RJ-obfuscation...')
        x_tr, x_val, y_tr, y_val = divide_dataset(texts, labels, only_train_val=True)
        x_te, y_te = [], []
    else:
        # dividing train-val-test
        print('Creating dataset RJ...')
        x_tr, x_val, x_te, y_tr, y_val, y_te = divide_dataset(texts, labels, only_train_val=False)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr)
    selected_authors = random.sample(np.unique(y_tr).tolist(), 10)
    x_tr, y_tr, _ = purge_authors(x_tr, y_tr, to_keep=selected_authors)
    x_val, y_val, _ = purge_authors(x_val, y_val, to_keep=selected_authors)
    if kwargs['obfuscation']:
        for file in os.listdir(data_path + 'test'):
            with open(data_path + 'test/' + file, 'r') as data_file:
                splits = get_splits(data_file.read())
                x_te.extend(splits)
                y_te.extend([file.split('.', 1)[0]] * len(splits))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


def process_pan11(data_path='../dataset/', **kwargs):
    data_path = data_path + 'pan11/'
    print('Creating dataset pan11...')
    train_path = data_path + 'pan11-author-identification-training-corpus-2011-04-08'
    x_tr, x_val, y_tr, y_val = [], [], [], []
    for file in os.listdir(train_path):
        if 'Verify' in file and 'GroundTruth' not in file:
            if 'Train' in file:
                texts, labels, _ = _fetch_xml(train_path + '/' + file, valid=False)
                for i, text in enumerate(texts):
                    splits = get_splits(text)
                    x_tr.extend(splits)
                    y_tr.extend([labels[i]] * len(splits))
            elif 'Valid' in file:
                texts, _, ids = _fetch_xml(train_path + '/' + file, valid=True)
                n = re.findall(r'\d+', file)[0]
                ground_truth = _fetch_groundtruth(train_path + f'/GroundTruthVerify{n}Valid+.xml')
                labels = [ground_truth[text_id] for text_id in ids]
                for i, text in enumerate(texts):
                    splits = get_splits(text)
                    x_val.extend(splits)
                    y_val.extend([labels[i]] * len(splits))
    x_tr, y_tr, to_keep = purge_authors(x_tr, y_tr)
    x_val, y_val, _ = purge_authors(x_val, y_val, to_keep)
    test_path = data_path + 'pan11-author-identification-test-corpus-2011-05-23'
    x_te, y_te = [], []
    for file in os.listdir(test_path):
        if 'Verify' in file and 'GroundTruth' not in file:
            texts, _, ids = _fetch_xml(test_path + '/' + file, valid=True)
            n = re.findall(r'\d+', file)[0]
            ground_truth = _fetch_groundtruth(test_path + f'/GroundTruthVerify{n}Test+.xml')
            labels = [ground_truth[text_id] for text_id in ids]
            for i, text in enumerate(texts):
                splits = get_splits(text)
                x_te.extend(splits)
                y_te.extend([labels[i]] * len(splits))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


def _fetch_xml(path, valid=False):
    # corpus = xml.etree.ElementTree.parse(path).getroot() # this doesn't work due to inconsistencies found within body elements
    lines = list(map(str.strip, open(path, 'rt').readlines()))
    texts, labels, text_ids = [], [], []
    rootname = 'training' if valid is False else 'testing'
    assert lines[0] == f'<{rootname}>', f'wrong root {lines[0]}'
    i = 1
    while lines[i] != f'</{rootname}>':
        if lines[i].startswith('<text file='):
            l = lines[i]
            text_ids.append(l[l.index('"') + 1:l.rindex('"')])
            i += 1
            author_id = None
            while lines[i] != '</text>':
                l = lines[i]
                if l.startswith('<author id='):
                    author_id = l[l.index('"') + 1:l.rindex('"')]
                elif l.startswith('<body>'):
                    i += 1
                    body = []
                    while lines[i] != '</body>':
                        if lines[i]:
                            body.append(lines[i])
                        i += 1
                    body = '\n'.join(body)
                i += 1
            texts.append(body)
            labels.append(author_id)
        i += 1
    return texts, labels, text_ids


def _fetch_groundtruth(path):
    ground_truth = {}
    results = xml.etree.ElementTree.parse(path).getroot()
    for result in results.findall('text'):
        file = result.attrib['file']
        file = file[file.find('"') + 1:]
        author = result.find('author').attrib['id']
        ground_truth[file] = author
    return ground_truth
