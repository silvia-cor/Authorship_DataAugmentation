import os
import pickle
from pathlib import Path

import torch
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import general.process_dataset
from classifiers.classifier_nn import NN_classification
from classifiers.classifier_svm import SVM_classification
from Training_Generation import TuningGeneration_pipeline
from general.helpers_text import labels_for_AV
from general.utils import pickled_resource, load_files, K_score
from general.visualization import plot_samples, save_generated_samples, plot_correlation_sample_performance
from general.significance_test import significance_test
import argparse

import nltk

nltk.download('punkt')
nltk.download('stopwords')


def run(dispatcher: dict, main_dir: str, classifier_name, generator_name, dataset_name, training_policy, device,
        gan_device):
    """
    :param dispatcher: dictionary-like dispatcher to run the correct classification process;
    :param main_dir: main directory for all the files;
    :param classifier_name: name of the classifier;
    :param generator_name: name of the generator;
    :param dataset_name: name of the dataset to process;
    :param training_policy: policy for training the generator (if None: no augmentation is performed);
    :param device: primary device on which to run the process (if multiple GPUs are available, it's the first one);
    :param gan_device: device(s) on which to run the gan processes (for multi-threading).
    """
    PICKLES_PATH = main_dir + 'pickles/'
    RESULTS_PATH = main_dir + 'results/'
    MODELS_PATH = main_dir + 'models/'
    DATA_PATH = main_dir + 'datasets/'

    dataset_path = PICKLES_PATH + f'pickled_data/dataset_{dataset_name}.pickle'
    pickle_path = PICKLES_PATH + f'preds_{dataset_name}.pickle'
    result_path = RESULTS_PATH + f'res_{dataset_name}.csv'
    plots_path = RESULTS_PATH + f'plots/{dataset_name}/SamplePlot_'
    gen_path = RESULTS_PATH + f'samples_{dataset_name}_{training_policy}_{generator_name}.txt'
    os.makedirs(str(Path(PICKLES_PATH)), exist_ok=True)
    os.makedirs(str(Path(RESULTS_PATH)), exist_ok=True)
    os.makedirs(str(Path(MODELS_PATH)), exist_ok=True)
    os.makedirs(str(Path(dataset_path).parent), exist_ok=True)
    os.makedirs(str(Path(plots_path).parent), exist_ok=True)
    os.makedirs(str(Path(MODELS_PATH + dataset_name)), exist_ok=True)

    if 'obf' in dataset_name:
        path = re.split('-', dataset_name)[0]
        obf = True
    else:
        path = dataset_name
        obf = False
    tr, val, te, labels_tr, labels_val, labels_te = \
        pickled_resource(dataset_path, getattr(general.process_dataset, f'process_{path}'), data_path=DATA_PATH, obfuscation=obf)
    print('\n# train samples:', len(labels_tr))
    print('# validation samples:', len(labels_val))
    print(f'# test samples: {len(labels_te)}\n')
    authors = np.unique(labels_tr)
    df_csv, df_preds = load_files(result_path, pickle_path)
    nn_type = re.split('(?<=.)(?=[A-Z])', generator_name)[0]
    method_name = classifier_name if classifier_name in ['svm', 'knn'] else classifier_name + f'_{nn_type}'
    base_name = method_name
    if training_policy is not None:
        method_name = classifier_name + f'_+{training_policy}' + f'_{generator_name}'
        if len(df_csv[df_csv['Method'] == base_name]) != len(authors):
            print('I first need the base classifier for comparison! Wait a bit...')
            run(dispatcher, main_dir, classifier_name, generator_name, dataset_name, training_policy=None,
                device=device, gan_device=gan_device)
            df_csv, df_preds = load_files(result_path, pickle_path)
        if classifier_name in ['svm', 'knn'] and training_policy == 'GANT' and \
                (len(df_csv[df_csv['Method'] == f'nn_+GANT_{generator_name}']) != len(authors)):
            print('I first need to run the nn experiment to employ the GAN Training! Wait a bit...')
            run(dispatcher, main_dir, 'nn', generator_name, dataset_name, training_policy='GANT',
                device=device, gan_device=gan_device)
            df_csv, df_preds = load_files(result_path, pickle_path)
    print(f'----- AV {method_name} experiment on {dataset_name}  -----')
    authors_metrics = []
    for author in authors:
        if author not in df_preds:
            df_preds[author] = {}
        if method_name in df_preds[author]:
            print(f'AV {method_name} experiment on {author} from {dataset_name} already done!')
        else:
            # we save the base nn classifier (only training on training set) for embedding etc. only without augmentation
            # we also save the final nn classifier (training also on validation set, with or without augmentation)
            nn_base_path = MODELS_PATH + f'{dataset_name}/nn_{nn_type}_base_{author}.pt'
            G_path = MODELS_PATH + f'{dataset_name}/{generator_name}_{training_policy}_{author}.pt'
            print(f'\n--- Author {author} ---')
            y_tr = labels_for_AV(labels_tr, author)
            y_val = labels_for_AV(labels_val, author)
            y_te = labels_for_AV(labels_te, author)
            if training_policy is None:
                generated_samples = []
                new_tr = tr
                new_y_tr = y_tr
                nn_final_path = MODELS_PATH + f'{dataset_name}/nn_{nn_type}_final_{author}.pt'
                nn_embed_path = None
            else:
                nn_final_path = MODELS_PATH + f'{dataset_name}/nn_+{training_policy}_{generator_name}_{author}.pt'
                author_samples = [tr[i] for i, label in enumerate(y_tr) if label == 1]
                generated_samples = TuningGeneration_pipeline(training_policy, author_samples, author, G_path,
                                                              nn_base_path, device, gan_device)
                new_tr = tr + generated_samples
                new_y_tr = y_tr + ([0] * len(generated_samples))
                if classifier_name == 'nn' and 'embed' not in generator_name:
                    save_generated_samples(gen_path, author_samples, generated_samples, dataset_name, author)
                nn_embed_path = nn_base_path
                nn_base_path = nn_final_path  # we do not need the base classifier with augmentation
            preds, targets, feats_extractor = dispatcher[classifier_name](new_tr, val, te, new_y_tr, y_val, y_te,
                                                                          device=device, gan_device=gan_device,
                                                                          generator_name=generator_name,
                                                                          nn_base_path=nn_base_path,
                                                                          nn_final_path=nn_final_path,
                                                                          nn_embed_path=nn_embed_path)
            if classifier_name == 'nn':
                print('Plotting the samples...')
                author_tr = [tr[i] for i, label in enumerate(y_tr) if label == 1]
                nonAuthor_tr = [tr[i] for i, label in enumerate(y_tr) if label == 0]
                author_te = [te[i] for i, label in enumerate(y_te) if label == 1]
                nonAuthor_te = [te[i] for i, label in enumerate(y_te) if label == 0]
                texts = author_tr + nonAuthor_tr + generated_samples + author_te + nonAuthor_te
                labels = [r'$A$ in training'] * len(author_tr) + [r'$\overline{A}$ in training'] * len(nonAuthor_tr) + \
                         [r'$Fake$ in training'] * len(generated_samples) + \
                         [r'$A$ in test'] * len(author_te) + [r'$\overline{A}$ in test'] * len(nonAuthor_te)
                plot_samples(texts, labels, classifier_name, feats_extractor,
                             plots_path + f'{method_name}_{author}.png', nn_final_path, device)
            if 'True' not in df_preds[author]:
                df_preds[author]['True'] = targets
            df_preds[author][method_name] = preds
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(df_preds, pickle_file)
        f1 = np.around(f1_score(df_preds[author]['True'], df_preds[author][method_name], average='binary',
                                zero_division=1), decimals=3)
        k = np.around(K_score(df_preds[author]['True'], df_preds[author][method_name]), decimals=3)
        print(f'F1: {f1}')
        print(f'K: {k}\n')
        authors_metrics.append((author, f1, k))
        df_csv = df_csv[df_csv.Method != method_name]
        for author_name, f1, k in authors_metrics:
            row = {'Method': method_name, 'Author_name': author_name, 'F1': f1, 'K': k}
            df_csv = pd.concat([df_csv, pd.DataFrame([row])], ignore_index=True)
        df_csv.to_csv(path_or_buf=result_path, sep=';', index=False, header=True)
    print(f'----- AV {method_name} experiment on {dataset_name} -- THE END -----')
    f1_mean_method = df_csv[df_csv['Method'] == method_name]['F1'].mean(axis=0)
    k_mean_method = df_csv[df_csv['Method'] == method_name]['K'].mean(axis=0)
    print(f'F1 mean: {f1_mean_method:.3f}')
    print(f'K mean: {k_mean_method:.3f}')
    if training_policy is not None:
        if base_name in df_csv['Method'].values:
            print(f'COMPARISON WITH BASELINE {base_name}')
            f1_mean_base = df_csv[df_csv['Method'] == base_name]['F1'].mean(axis=0)
            k_mean_base = df_csv[df_csv['Method'] == base_name]['K'].mean(axis=0)
            delta_f1 = (f1_mean_method - f1_mean_base) / f1_mean_base * 100
            delta_k = (k_mean_method - k_mean_base) / k_mean_base * 100
            print(f'F1 mean Delta %: {delta_f1:.2f}')
            print(f'K mean Delta %: {delta_k:.2f}')
            significance_test(pickle_path, authors, method_name, base_name)
        else:
            print(f'No {base_name} saved, significance test cannot be performed :/\n')
    else:
        print('No significance test requested\n')


# the generator is important even if there is no augmentation (i.e., no training_policy) because it determines the nn architecture
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment of data augmentation for Authorship Verification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('-classifier', type=str, required=True, help='Name of the learner to train as classifier')
    parser.add_argument('--generator', type=str, required=False, default='gpt2', help='Name of the learner to train as generator')
    parser.add_argument('--training_policy', type=str, required=False, default=None,
                        help='Training policy for the generator (if left to default None, augmentation is not performed)')
    parser.add_argument('--devices', help='GPU device(s) if available, e.g.: "0,1,2"', required=False,
                        type=str, default='0')
    parser.add_argument('--main_dir', type=str, required=False, default='../',
                        help='Path to the main directory for all the files, both for input and output')
    args = parser.parse_args()

    gan_device = [int(item) for item in args.devices.split(',')]
    device = torch.device(gan_device[0] if torch.cuda.is_available() else 'cpu')
    assert args.dataset in ['victoria', 'tweepfake', 'ebg', 'ebg-obf', 'rj', 'rj-obf', 'pan11'], \
        'This dataset is not available. Options: victoria, tweepfake, ebg, ebg-obf, rj, rj-obf, pan11.'
    assert args.classifier in ['svm', 'nn'], \
        'This classifier is not implemented. Options: svm, nn.'
    assert args.generator in ['gpt2', 'onehotTrans', 'embedTrans', 'onehotGru', 'embedGru'], \
        'This generator is not implemented. Options: gpt2, onehotTrans, embedTrans, onehotGru, embedGru.'
    assert args.training_policy in [None, 'LMT', 'GANT'], \
        'This training policy is not implemented. Options: None (no augmentation), LMT, GANT.'
    assert not (args.classifier in ['svm', 'knn'] and "embed" in args.generator and args.training_policy is not None), \
        "The svm learner cannot process word-embedding samples."

    print(f'----- CONFIGURATION -----')
    d = vars(args)
    for k in d:
        print(k, ':', d[k])
    print('(Primary) device:', device)
    print('\n')

    dispatcher = {'svm': SVM_classification, 'nn': NN_classification}

    run(dispatcher, args.main_dir, args.classifier, args.generator, args.dataset, args.training_policy, device,
        gan_device)
