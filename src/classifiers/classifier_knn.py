import sys

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.sparse import vstack

from BaseFeatures_extractor import featuresExtractor

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


def KNN_classification(tr, val, te, y_tr, y_val, y_te, **kwargs):
    """
    Perform classification via K-NN with hyperparameters optimization (via GridSearch).
    """
    params = {'n_neighbors': [1, 2, 3, 5, 10, 30, 50]}
    print('Features extraction (train/val)...')
    X_tr, X_val, _ = feat_extraction(tr, val, y_tr)
    print('Train shape:', X_tr.shape)
    print('Validation shape:', X_val.shape)
    print('GridSearch...')
    knn = KNeighborsClassifier()
    train_indices = np.full((len(y_tr),), -1, dtype=int)
    val_indices = np.full((len(y_val),), 0, dtype=int)
    ps = PredefinedSplit(np.append(train_indices, val_indices))
    f1 = make_scorer(f1_score, average='binary', zero_division=1)
    grid_search = GridSearchCV(knn, params, cv=ps, scoring=f1, refit=False, n_jobs=4, verbose=2)
    X = vstack((X_tr, X_val))
    y_trval = y_tr + y_val
    grid_search.fit(X, y_trval)
    best_knn_params = grid_search.cv_results_['params'][grid_search.best_index_]
    print('Best model:', best_knn_params)
    best_knn = KNeighborsClassifier(**best_knn_params)
    print('Features extraction (trval/test)...')
    trval = tr + val
    X_trval, X_te, feat_extractor = feat_extraction(trval, te, y_trval)
    print('Train+val shape:', X_trval.shape)
    print('Test shape:', X_te.shape)
    print('Testing the K-NN...')
    best_knn.fit(X_trval, y_trval)
    preds = best_knn.predict(X_te)
    return preds, y_te, feat_extractor


# extract the TfIdf of the char n-grams
def feat_extraction(x_tr, x_te, y_tr, feat_ratio=0.1):
    bf_extractor = featuresExtractor(x_tr)
    feat_extractor = {'bf_extractor': bf_extractor}
    X_tr = bf_extractor.extract(x_tr)
    X_te = bf_extractor.extract(x_te)
    #vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), sublinear_tf=True)
    #f_tr = vectorizer.fit_transform(x_tr)
    #f_te = vectorizer.transform(x_te)
    #feat_extractor['vectorizer'] = vectorizer
    #if feat_ratio != 1:
    #    num_feats = int(f_tr.shape[1] * feat_ratio)
    #    selector = SelectKBest(chi2, k=num_feats)
    #    f_tr = selector.fit_transform(f_tr, y_tr)
    #    f_te = selector.transform(f_te)
    #    feat_extractor['selector'] = selector
    #X_tr = hstack((X_tr, f_tr))
    #X_te = hstack((X_te, f_te))
    return X_tr, X_te, feat_extractor
