import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from scipy.sparse import coo_matrix, hstack
import pymorphy2

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from utils import *
from preparations import *


def train_and_test():
    sites, test_sites = load_sites(WEBPAGES_PATH)

    text_vectorizer = CountVectorizer()
    attrs_vectorizer = CountVectorizer()
    model = LogisticRegression()

    text_data, attrs_data = [], []

    failsFile = open(RESULTS_PATH + 'fails.txt', 'w')
    for site in sites:
        try:
            text, attrs = load_file(site[0])
        except Exception as exception:
            failsFile.write('{0}: {1}\n'.format(site[1], exception))
            continue
        text_data.append(text)
        attrs_data.append(attrs)

    write_to_file(RESULTS_PATH, 'text_data.txt', json.dumps(text_data))
    write_to_file(RESULTS_PATH, 'attrs_data.txt', json.dumps(attrs_data))

    a = text_vectorizer.fit_transform(text_data)
    b = attrs_vectorizer.fit_transform(attrs_data)

    # print(type(a))
    # print(type(b))
    # print(a.shape)
    # print(b.shape)

    X, y, fails = corpus_transformation(sites, text_vectorizer, attrs_vectorizer)

    # kf = KFold(n_splits=10)
    # print('CROSS VALIDATION')
    # i = 1
    # for train, test in kf.split(X):
    model.fit(X, y)

    # kf = KFold(n_splits=10)
    # print('CROSS VALIDATION')
    # i = 1
    # for train, test in kf.split(X):
    #     model.fit(X[train], y[train])
    #     prediction = model.predict(X[test])
    #     print('Number ot iteration: ' + str(i))
    #     print(metrics.classification_report(y[test], prediction))
    #     i += 1

    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print('-'*60)
    print('Results of training:')
    print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    X_test, y_test, fails = corpus_transformation(test_sites, text_vectorizer, attrs_vectorizer)
    failsFile.write(fails)
    failsFile.close()
    predicted = model.predict(X_test)
    print('Results of testing:')
    print(metrics.classification_report(y_test, predicted))
    # print(metrics.confusion_matrix(y_test, predicted))


if __name__ == '__main__':
    train_and_test()
