import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar


# convert the predictions into binary values (1 = correct author ; 0 = wrong author)
def _convert_preds(y_true, y_pred):
    result = []
    for true,pred in zip(y_true, y_pred):
        res = 1 if true == pred else 0
        result.append(res)
    return np.array(result)


# prepare the contingency table for the mcnemar's test
def _prepare_mcnemar_table(a,b):
    y_y= sum((a == 1) * (b == 1))
    y_n= sum((a == 1) * (b == 0))
    n_y= sum((a == 0) * (b == 1))
    n_n= sum((a == 0) * (b == 0))
    table = [[y_y,y_n],[n_y,n_n]]
    return table


def _prepare_predictions(pickle_path, authors, method_name, baseline_name):
    with open(pickle_path, 'rb') as pickle_file:
        df_preds = pickle.load(pickle_file)
    y_true, y_baseline, y_method = [], [], []
    for author in authors:
        y_true.extend(df_preds[author]['True'])
        y_baseline.extend(df_preds[author][baseline_name])
        y_method.extend(df_preds[author][method_name])
    return y_true, y_baseline, y_method


def significance_test(pickle_path, authors, method_name, baseline_name):
    y_true, y_baseline, y_method = _prepare_predictions(pickle_path, authors, method_name, baseline_name)
    print(f'----- MCNEMAR TEST AGAINST {baseline_name} -----')
    y_baseline_conv = _convert_preds(y_true, y_baseline)
    y_method_conv = _convert_preds(y_true, y_method)
    test_table = _prepare_mcnemar_table(y_baseline_conv, y_method_conv)
    test_result = mcnemar(test_table)
    stat, p = test_result.statistic, test_result.pvalue
    print(f'Statistics= {stat:.3f}')
    print(f'p= {p:.3f}')
    alpha = 0.05
    if p > alpha:
        print('Same proportion (difference is not significant)\n')
    else:
        print('Different proportion (difference is significant)\n')


def compute_correlation(labels_tr, df_csv, methods):
    authors = np.unique(labels_tr)
    authors_values = []
    for author in authors:
        author_value = [len([label for label in labels_tr if label == author])]
        author_value.extend(
            np.array([df_csv[(df_csv['Method'] == method_name) & (df_csv['Author_name'] == author)]['F1']
                      for method_name in methods]).flatten())
        authors_values.append(author_value)
    cols = ['tr']
    methods = [method.replace('+', '') for method in methods]
    cols.extend(methods)
    df = pd.DataFrame(authors_values, columns=cols)
    z = df.pop('tr')
    model = sm.OLS(z, sm.add_constant(df)).fit()
    print(model.summary())
    return model, methods













