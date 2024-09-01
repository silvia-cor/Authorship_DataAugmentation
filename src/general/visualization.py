import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.sparse import hstack
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

import general.process_dataset
from general.dataloader import CustomDataloader
from general.significance_test import compute_correlation
from general.utils import pickled_resource, load_files

random.seed(42)


# plot the dataset samples and the centroids
def plot_samples(samples, labels, learner_name, feats_extractor, save_path, nn_path, device):
    palette = {r'$A$ in training': 'forestgreen',
               r'$\overline{A}$ in training': 'royalblue',
               r'$Fake$ in training': 'darkorange',
               r'$A$ in test': 'palegreen',
               r'$\overline{A}$ in test': 'lightskyblue'}
    if learner_name == 'svm':
        BFeats = feats_extractor['bf_extractor'].extract(samples)
        feats = feats_extractor['vectorizer'].transform(samples)
        if 'selector' in feats_extractor:
            feats = feats_extractor['selector'].transform(feats)
        feats = hstack((BFeats, feats))
    elif learner_name == 'nn':
        onehot = True if "onehot" in nn_path else False
        nn_path = None if "onehot" in nn_path else nn_path
        dataloader = CustomDataloader([samples, labels], feats_extractor, batch_size=8,
                                      shuffle=False, labels_float=False, onehot=onehot,
                                      nn_embed_path=nn_path).dataloader
        feats, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                inp = batch[0].to(device)
                bf = batch[1] if feats_extractor.bf_extractor is None else batch[1].to(device)
                feats.append(feats_extractor.model.representation(inp, bf).detach().clone().cpu())
                labels.extend(batch[2])
        feats = torch.cat(feats, dim=0)
    else:
        feats = feats_extractor.encode([samples, labels])
        feats = torch.stack(feats, dim=0)
    tsne = np.array(TSNE().fit_transform(feats))
    scatterplot = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=labels, palette=palette,
                                  hue_order=[r'$A$ in training', r'$\overline{A}$ in training', r'$Fake$ in training',
                                             r'$A$ in test',
                                             r'$\overline{A}$ in test'])
    if r'$Fake$ in training' in labels:
        # computing and plotting the centroids
        fake_tsne = tsne[[i for i, label in enumerate(labels) if label == r'$Fake$ in training']]
        A_tsne = tsne[[i for i, label in enumerate(labels) if label == r'$A$ in training' or label == r'$A$ in test']]
        notA_tsne = tsne[[i for i, label in enumerate(labels) if label == r'$\overline{A}$ in training'
                          or label == r'$\overline{A}$ in test']]
        notA_fake_tsne = np.vstack((notA_tsne, fake_tsne))
        notA_centroid = _compute_centroid(notA_tsne)
        A_centroid = _compute_centroid(A_tsne)
        notA_fake_centroid = _compute_centroid(notA_fake_tsne)
        sns.scatterplot(x=A_centroid[[0]], y=A_centroid[[1]], s=100, c='darkslategrey', label=r'$A$ centroid',
                        marker='X')
        sns.scatterplot(x=[notA_centroid[0]], y=[notA_centroid[1]], s=100, c='blue', label=r'$\overline{A}$ centroid',
                        marker='X')
        sns.scatterplot(x=notA_fake_centroid[[0]], y=notA_fake_centroid[[1]], s=100, c='blueviolet',
                        label=r'$\overline{A}$+$Fake$ centroid', marker='X')
        centroid_pairs = [(notA_centroid, A_centroid), (notA_fake_centroid, A_centroid)]
        for centroid1, centroid2 in centroid_pairs:
            dist = cosine(centroid1, centroid2)
            x_values = [centroid1[0], centroid2[0]]
            y_values = [centroid1[1], centroid2[1]]
            scatterplot.plot(x_values, y_values, color='gray', linestyle='--', alpha=0.5)
            mid_x = (centroid1[0] + centroid2[0]) / 2
            mid_y = (centroid1[1] + centroid2[1]) / 2
            scatterplot.text(mid_x, mid_y, f'{dist:.2f}', fontsize=9, color='black', ha='center')
    plt.legend(markerscale=.8, fontsize=9)
    plt.savefig(save_path)
    plt.cla()


def save_generated_samples(gen_path, author_samples, generated_texts, dataset_name, author):
    with open(gen_path, 'a') as f:
        f.write(f'Real texts for {dataset_name}_{author}\n')
        selected_author_samples = random.sample(author_samples, 5)
        for selected_author_sample in selected_author_samples:
            f.write(selected_author_sample + '\n')
    with open(gen_path, 'a') as f:
        f.write(f'Generation for {dataset_name}_{author}\n')
        selected_generated_texts = random.sample(generated_texts, 5)
        for selected_generated_text in selected_generated_texts:
            f.write(selected_generated_text + '\n')
    with open(gen_path, 'a') as f:
        f.write('\n\n')


# plot the correlation among the performance results and the length of the authors' production
def plot_correlation_sample_performance(dataset_path, dataset_name, result_path, pickle_path):
    tr, val, te, labels_tr, labels_val, labels_te = \
        pickled_resource(dataset_path, getattr(general.process_dataset, f'process_{dataset_name}'),
                         obfuscation=True)
    df_csv, df_preds = load_files(result_path, pickle_path)
    methods = np.unique(df_csv['Method'])
    print(f'Correlation for dataset {dataset_name} with svm methods...')
    methods_compute = [method for method in methods if 'svm' in method]
    _plot_corr(labels_tr, df_csv, methods_compute)
    print(f'Correlation for dataset {dataset_name} with nn_gpt2 methods...')
    methods_compute = [method for method in methods if 'nn' in method and 'gpt2' in method]
    _plot_corr(labels_tr, df_csv, methods_compute)
    print(f'Correlation for dataset {dataset_name} with nn_onehot methods...')
    methods_compute = [method for method in methods if 'nn' in method and 'onehot' in method]
    _plot_corr(labels_tr, df_csv, methods_compute)
    print(f'Correlation for dataset {dataset_name} with nn_embed methods...')
    methods_compute = [method for method in methods if 'nn' in method and 'embed' in method]
    _plot_corr(labels_tr, df_csv, methods_compute)


def _plot_corr(labels_tr, df_csv, methods_compute):
    osl, methods_st = compute_correlation(labels_tr, df_csv, methods_compute)
    f1s, p_values = [], []
    for i, method in enumerate(methods_compute):
        f1s.append(df_csv[df_csv['Method'] == method]['F1'].mean(axis=0))
        p_values.append(osl.pvalues.loc[methods_st[i]])
    types = (['f1_mean'] * len(methods_compute)) + (['p_values'] * len(methods_compute))
    methods_compute = [method_compute.replace('_', '_\n') for method_compute in methods_compute]
    mfp = list(zip(methods_compute, f1s, p_values))
    methods_compute, f1s, p_values = zip(*sorted(mfp, key=lambda x: x[1]))
    df = pd.DataFrame({'methods': methods_compute + methods_compute, 'values': f1s + p_values, 'type': types})
    sns.barplot(data=df, x="methods", y="values", hue='type')
    # plt.xticks(rotation=60)
    plt.tick_params(labelsize=7)
    plt.show()


# create the Latex coloured table of the results
def colorize_table(values, maxtone=50):
    ten_percent = int(0.1 * len(values))
    pos_values = [val for val in values if val >= 0]
    neg_values = [val for val in values if val < 0]
    lowest_values = [val for val in sorted(neg_values)[:ten_percent]]
    highest_values = [val for val in sorted(pos_values, reverse=True)[:ten_percent]]
    clip_neg = max(lowest_values)
    clip_pos = min(highest_values)
    for val in values:
        if val >= 0:
            color = 'green'
            clipped_val = max(0, min(clip_pos, val))
            tone = maxtone * (clipped_val - 0) / (clip_pos - 0)
        else:
            color = 'red'
            clipped_val = max(clip_neg, min(0, val))
            tone = maxtone * ((0 - clipped_val) / (0 - clip_neg))
        s = '\cellcolor{' + color + f'!{int(tone)}' + '}'
        print(val, s)


def _compute_centroid(samples):
    return np.mean(samples, axis=0)
