import os
import pickle
import random
from pathlib import Path

import pandas as pd
import torch
import re

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import general.process_dataset
from classifiers.classifier_nn import NN_classification
from classifiers.classifier_svm import SVM_classification
from classifiers.classifier_bert import Bert_classification
from Training_Generation import TuningGeneration_pipeline
from general.helpers_text import labels_for_AV
from general.utils import pickled_resource, load_files
from general.visualization import plot_samples, save_generated_samples, plot_correlation_sample_performance
from general.significance_test import significance_test


def process(dispatcher, learner_name, generator_name, dataset_name,
            dataset_path, pickle_path, result_path, plots_path, gen_path, training_policy, device):
    if 'obf' in dataset_name:
        path = re.split('-', dataset_name)[0]
        obf = True
    else:
        path = dataset_name
        obf = False
    tr, val, te, labels_tr, labels_val, labels_te = \
        pickled_resource(dataset_path, getattr(general.process_dataset, f'process_{path}'), obfuscation=obf)
    print('\n# train samples:', len(labels_tr))
    print('# validation samples:', len(labels_val))
    print(f'# test samples: {len(labels_te)}\n')
    authors = np.unique(labels_tr)
    df_csv, df_preds = load_files(result_path, pickle_path)
    nn_type = re.split('(?<=.)(?=[A-Z])', generator_name)[0]
    method_name = learner_name if learner_name in ['svm', 'bert'] else learner_name + f'_{nn_type}'
    base_name = method_name
    if training_policy is not None:
        method_name = learner_name + f'_+{training_policy}' + f'_{generator_name}'
        if len(df_csv[df_csv['Method'] == base_name]) != len(authors):
            print('I first need the base classifier for comparison! Wait a bit...')
            process(dispatcher, learner_name, generator_name, dataset_name,
                    dataset_path, pickle_path, result_path, plots_path, None, training_policy=None, device=device)
            df_csv, df_preds = load_files(result_path, pickle_path)
        if learner_name == 'svm' and training_policy == 'GANT' and \
                (len(df_csv[df_csv['Method'] == f'nn_+GANT_{generator_name}']) != len(authors)):
            print('I first need to run the nn experiment to employ the GAN Training! Wait a bit...')
            process(dispatcher, 'nn', generator_name, dataset_name,
                    dataset_path, pickle_path, result_path, plots_path, gen_path, training_policy='GANT', device=device)
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
            nn_base_path = f'../models2/{dataset_name}/nn_{nn_type}_base_{author}.pt'
            G_path = f'../models2/{dataset_name}/{generator_name}_{training_policy}_{author}.pt'
            print(f'\n--- Author {author} ---')
            y_tr = labels_for_AV(labels_tr, author)
            y_val = labels_for_AV(labels_val, author)
            y_te = labels_for_AV(labels_te, author)
            if training_policy is None:
                generated_samples = []
                new_tr = tr
                new_y_tr = y_tr
                nn_final_path = f'../models2/{dataset_name}/nn_{nn_type}_final_{author}.pt'
                nn_embed_path = None
            else:
                nn_final_path = f'../models2/{dataset_name}/nn_+{training_policy}_{generator_name}_{author}.pt'
                author_samples = [tr[i] for i, label in enumerate(y_tr) if label == 1]
                negative_samples = [tr[i] for i, label in enumerate(y_tr) if label == 0]
                generated_samples = TuningGeneration_pipeline(training_policy, tr, y_tr, author_samples, author,
                                                              len(negative_samples), G_path, nn_base_path, device)
                # sel_negative_samples = random.sample(negative_samples,
                #                                     len(negative_samples) - len(
                #                                         generated_samples))  # without replacement
                # new_tr = author_samples + generated_samples + sel_negative_samples
                new_tr = tr + generated_samples
                # new_y_tr = ([1] * len(author_samples)) + ([0] * (len(generated_samples) + len(sel_negative_samples)))
                new_y_tr = y_tr + ([0] * len(generated_samples))
                if learner_name == 'nn' and 'embed' not in generator_name:
                    save_generated_samples(gen_path, author_samples, generated_samples, dataset_name, author)
                nn_embed_path = nn_base_path
                nn_base_path = nn_final_path  # we do not need the base classifier with augmentation
            preds, targets, feats_extractor = dispatcher[learner_name](new_tr, val, te, new_y_tr, y_val, y_te,
                                                                       device=device,
                                                                       generator_name=generator_name,
                                                                       nn_base_path=nn_base_path,
                                                                       nn_final_path=nn_final_path,
                                                                       nn_embed_path=nn_embed_path)
            if learner_name in ['nn', 'bert']:
                print('Plotting the samples...')
                author_tr = [tr[i] for i, label in enumerate(y_tr) if label == 1]
                nonAuthor_tr = [tr[i] for i, label in enumerate(y_tr) if label == 0]
                author_te = [te[i] for i, label in enumerate(y_te) if label == 1]
                nonAuthor_te = [te[i] for i, label in enumerate(y_te) if label == 0]
                texts = author_tr + nonAuthor_tr + generated_samples + author_te + nonAuthor_te
                labels = ['Author_tr'] * len(author_tr) + ['NonAuthor_tr'] * len(nonAuthor_tr) + \
                         ['Fake'] * len(generated_samples) + \
                         ['Author_te'] * len(author_te) + ['NonAuthor_te'] * len(nonAuthor_te)
                plot_samples(texts, labels, learner_name, feats_extractor,
                             plots_path + f'{method_name}_{author}.png', nn_final_path, device)
            if 'True' not in df_preds[author]:
                df_preds[author]['True'] = targets
            df_preds[author][method_name] = preds
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(df_preds, pickle_file)
        f1 = np.around(f1_score(df_preds[author]['True'], df_preds[author][method_name], average='binary',
                                zero_division=1), decimals=3)
        acc = np.around(accuracy_score(df_preds[author]['True'], df_preds[author][method_name]), decimals=3)
        print(f'F1: {f1}')
        print(f'Acc: {acc}\n')
        authors_metrics.append((author, f1, acc))
        df_csv = df_csv[df_csv.Method != method_name]
        for author_name, f1, acc in authors_metrics:
            row = {'Method': method_name, 'Author_name': author_name, 'F1': f1, 'Acc': acc}
            df_csv = df_csv.append(row, ignore_index=True)
        df_csv.to_csv(path_or_buf=result_path, sep=';', index=False, header=True)
    print('----- THE END -----')
    f1_mean_method = df_csv[df_csv['Method'] == method_name]['F1'].mean(axis=0)
    acc_mean_method = df_csv[df_csv['Method'] == method_name]['Acc'].mean(axis=0)
    print(f'F1 mean: {f1_mean_method:.3f}')
    print(f'Acc mean: {acc_mean_method:.3f}')
    if training_policy is not None:
        if base_name in df_csv['Method'].values:
            print(f'COMPARISON WITH BASELINE {base_name}')
            f1_mean_base = df_csv[df_csv['Method'] == base_name]['F1'].mean(axis=0)
            acc_mean_base = df_csv[df_csv['Method'] == base_name]['Acc'].mean(axis=0)
            delta_f1 = (f1_mean_method - f1_mean_base) / f1_mean_base * 100
            delta_acc = (acc_mean_method - acc_mean_base) / acc_mean_base * 100
            print(f'F1 mean Delta %: {delta_f1:.2f}')
            print(f'Acc mean Delta %: {delta_acc:.2f}')
            significance_test(pickle_path, authors, method_name, base_name)
        else:
            print(f'No {base_name} saved, significance test cannot be performed :/\n')
    else:
        print('No significance test requested\n')


# the generator_name is potentially important even if there is no augmentation (i.e., no training_policy)
# because, given a certain generator, there is an ad-hoc nn architecture
if __name__ == "__main__":
    def main():
        dataset_name = 'victoria'
        learner_name = 'svm'
        generator_name = 'onehotTrans'
        training_policy = 'GANT'  # if None, augmentation is not applied
        device = torch.device(6 if torch.cuda.is_available() else 'cpu')
        assert dataset_name in ['victoria', 'tweepfake', 'ebg', 'ebg-obf', 'rj', 'rj-obf', 'pan11'], \
            'This dataset is not available. Options: victoria, tweepfake, ebg, ebg-obf, rj, rj-obf, pan11.'
        assert learner_name in ['svm', 'nn', 'bert'], \
            'This learner is not implemented. Options: svm, nn, bert.'
        assert generator_name in ['gpt2', 'onehotTrans', 'embedTrans', 'onehotGru', 'embedGru'], \
            'This generator is not implemented. Options: gpt2, onehotTrans, embedTrans, onehotGru, embedGru.'
        assert training_policy in [None, 'LMT', 'GANT'], \
            'This training policy is not implemented. Options: None, LMT, GANT.'
        assert not (learner_name == 'svm' and "embed" in generator_name and training_policy is not None), \
            "The svm learner cannot process word-embedding samples."

        dispatcher = {'svm': SVM_classification, 'nn': NN_classification, 'bert': Bert_classification}

        dataset_path = f'../pickles2/data/dataset_{dataset_name}.pickle'
        pickle_path = f'../pickles2/preds_{dataset_name}.pickle'
        result_path = f'../results2/res_{dataset_name}.csv'
        plots_path = f'../results2/plots/{dataset_name}/SamplePlot_'
        gen_path = f'../results2/samples_{dataset_name}_{training_policy}_{generator_name}.txt'
        os.makedirs(str(Path(dataset_path).parent), exist_ok=True)
        os.makedirs(str(Path(plots_path).parent), exist_ok=True)
        os.makedirs(str(Path(f'../models2/{dataset_name}')), exist_ok=True)

        process(dispatcher, learner_name, generator_name, dataset_name,
                dataset_path, pickle_path, result_path, plots_path, gen_path, training_policy, device)


    main()
