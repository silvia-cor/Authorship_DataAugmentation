import sys
from functools import partial
from multiprocessing import Pool

import tqdm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

from BaseFeatures_extractor import featuresExtractor

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


def SVM_classification(tr, val, te, y_tr, y_val, y_te, **kwargs):
    """
    Perform classification via SVM with hyperparameters optimization (via GridSearch).
    """
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'class_weight': ['balanced'],
              'random_state': [42]}
    print('Features extraction (train/val)...')
    X_tr, X_val, _ = feat_extraction(tr, val, y_tr)
    print('Train shape:', X_tr.shape)
    print('Validation shape:', X_val.shape)
    print('GridSearch...')
    param_grid = ParameterGrid(params)
    with Pool(processes=24) as p:
        gridsearch_step = partial(_gridSearch_step, X_tr, X_val, y_tr, y_val)
        gridsearch_results = list(tqdm.tqdm(p.imap(gridsearch_step, param_grid), total=len(param_grid)))
    best_result_idx = gridsearch_results.index(max(gridsearch_results, key=lambda result: result))
    print('Best model:', param_grid[best_result_idx])
    best_svm = SVC(**param_grid[best_result_idx])
    y_trval = y_tr + y_val
    print('Features extraction (trval/test)...')
    trval = tr + val
    X_trval, X_te, feat_extractor = feat_extraction(trval, te, y_trval)
    print('Train+val shape:', X_trval.shape)
    print('Test shape:', X_te.shape)
    print('Testing the SVM...')
    best_svm.fit(X_trval, y_trval)
    preds = best_svm.predict(X_te)
    return preds, y_te, feat_extractor


# extract the TfIdf of the char n-grams
def feat_extraction(x_tr, x_te, y_tr, feat_ratio=0.1):
    bf_extractor = featuresExtractor(x_tr)
    feat_extractor = {'bf_extractor': bf_extractor}
    X_tr = bf_extractor.extract(x_tr)
    X_te = bf_extractor.extract(x_te)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), sublinear_tf=True)
    f_tr = vectorizer.fit_transform(x_tr)
    f_te = vectorizer.transform(x_te)
    feat_extractor['vectorizer'] = vectorizer
    if feat_ratio != 1:
        num_feats = int(f_tr.shape[1] * feat_ratio)
        selector = SelectKBest(chi2, k=num_feats)
        f_tr = selector.fit_transform(f_tr, y_tr)
        f_te = selector.transform(f_te)
        feat_extractor['selector'] = selector
    X_tr = hstack((X_tr, f_tr))
    X_te = hstack((X_te, f_te))
    return X_tr, X_te, feat_extractor


# perform a single experiment of the GridSearch process
def _gridSearch_step(X_tr, X_val, y_tr, y_val, params):
    svm = SVC(**params)
    svm.fit(X_tr, y_tr)
    preds = svm.predict(X_val)
    f1 = f1_score(y_val, preds, average='binary', zero_division=1)
    return f1
