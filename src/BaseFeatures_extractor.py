from collections import Counter

import nltk
import spacy
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from general.helpers_text import tokenize_nopunct


# ------------------------------------------------------------------------
# Feature Extractor
# ------------------------------------------------------------------------

class featuresExtractor:
    def __init__(self, tr_docs):
        self.fw = stopwords.words('english')
        lens = [len(word) for word in tokenize_nopunct(' '.join(tr_docs))]
        self.words_upto = max([k for k, v in Counter(lens).items() if v >= 5])
        self.pos_tagger = spacy.load('en_core_web_sm')
        tr_pos_tags = [' '.join([word.tag_ for word in self.pos_tagger(doc)]) for doc in tr_docs]
        pos_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b",
                                         sublinear_tf=True)
        self.pos_vectorizer = pos_vectorizer.fit(tr_pos_tags)

    def extract(self, docs, verbose=False):
        # final matrixes of features
        # initialize the right number of rows, or hstack won't work
        X = csr_matrix((len(docs), 0))

        f = normalize(self._function_words_freq(docs))
        X = hstack((X, f))
        if verbose:
            print(f'task function words (#features={f.shape[1]}) [Done]')

        f = normalize(self._word_lengths_freq(docs))
        X = hstack((X, f))
        if verbose:
            print(f'task word lengths (#features={f.shape[1]}) [Done]')

        pos_tags = [' '.join([word.tag_ for word in self.pos_tagger(doc)]) for doc in docs]
        f = self.pos_vectorizer.transform(pos_tags)
        X = hstack((X, f))
        if verbose:
            print(f'task pos-tags (#features={f.shape[1]}) [Done]')
        return X

    # extract the frequency (L1x1000) of each function word used in the documents
    def _function_words_freq(self, documents):
        features = []
        for text in documents:
            mod_tokens = tokenize_nopunct(text)
            freqs = nltk.FreqDist(mod_tokens)
            nwords = len(mod_tokens)
            if nwords == 0:
                funct_words_freq = [float(0)] * len(self.fw)
            else:
                funct_words_freq = [freqs[function_word] / nwords for function_word in self.fw]
            features.append(funct_words_freq)
        f = csr_matrix(features)
        return f

    # extract the frequencies (L1x1000) of the words' lengths used in the documents,
    # following the idea behind Mendenhall's Characteristic Curve of Composition
    def _word_lengths_freq(self, documents):
        features = []
        for text in documents:
            mod_tokens = tokenize_nopunct(text)
            nwords = len(mod_tokens)
            if nwords == 0:
                tokens_count = [float(0)] * self.words_upto
            else:
                tokens_len = [len(token) for token in mod_tokens]
                tokens_count = []
                for i in range(1, self.words_upto + 1):
                    tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
            features.append(tokens_count)
        f = csr_matrix(features)
        return f
