import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from general.dataloader import CustomDataloader, BertDataloader
from scipy.sparse import hstack
import random
from general.significance_test import compute_correlation
from general.utils import pickled_resource, load_files
import general.process_dataset
import numpy as np
import pandas as pd

random.seed(42)


def plot_samples(samples, labels, learner_name, feats_extractor, save_path, nn_path, device):
    palette = {'Author_tr': 'forestgreen',
               'NonAuthor_tr': 'royalblue',
               'Fake': 'darkorange',
               'Author_te': 'palegreen',
               'NonAuthor_te': 'lightskyblue'}
    if learner_name == 'svm':
        BFeats = feats_extractor['bf_extractor'].extract(samples)
        feats = feats_extractor['vectorizer'].transform(samples)
        if 'selector' in feats_extractor:
            feats = feats_extractor['selector'].transform(feats)
        feats = hstack((BFeats, feats))
    elif learner_name == 'nn':
        onehot = True if "onehot" in nn_path else False
        dataloader = CustomDataloader([samples, labels], feats_extractor, batch_size=8,
                                      shuffle=False, labels_long=False, onehot=onehot, nn_embed_path=nn_path).dataloader
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
    tsne = TSNE().fit_transform(feats)
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=labels, palette=palette,
                    hue_order=['Author_tr', 'NonAuthor_tr', 'Fake', 'Author_te', 'NonAuthor_te'])
    plt.legend()
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

