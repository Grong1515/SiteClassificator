import codecs
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
    model = [LogisticRegression() for i in range(len(CLASSES_LIST))]

    text_data, attrs_data = [], []

    failsFile = codecs.open(RESULTS_PATH + 'fails.txt', mode='w', encoding='utf-8', errors='ignore')
    for site in sites:
        try:
            text, attrs = load_file(site[0])
        except Exception as exception:
            failsFile.write('{0}: {1}\n'.format(site[1], exception))
            continue
        text_data.append(text)
        attrs_data.append(attrs)

    """with codecs.open(os.path.join(os.path.normpath(RESULTS_PATH), 'text_data.txt'), mode='w', encoding='utf-8') as fp:
        json.dump(text_data, fp, ensure_ascii=False, indent=4)
    with codecs.open(os.path.join(os.path.normpath(RESULTS_PATH), 'attrs_data.txt'), mode='w', encoding='utf-8') as fp:
        json.dump(attrs_data, fp, ensure_ascii=False, indent=4)"""
    write_to_file(RESULTS_PATH, 'text_data.txt', json.dumps(text_data, ensure_ascii=False, indent=4))
    write_to_file(RESULTS_PATH, 'attrs_data.txt', json.dumps(attrs_data, ensure_ascii=False, indent=4))

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
    # for model in models:
    for i in range(len(CLASSES_LIST)):
        model[i].fit(X, y[i])

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
    f = codecs.open('results.txt', mode='w', encoding='utf-8', errors='ignore')
    print('Results of training:', file=f)
    for i in range(len(CLASSES_LIST)):
        expected = y[i]
        predicted = model[i].predict(X)
        print(CLASSES_LIST[i] + ':', file=f)
        print(metrics.classification_report(expected, predicted), file=f)
    # print(metrics.confusion_matrix(expected, predicted))

    X_test, y_test, fails = corpus_transformation(test_sites, text_vectorizer, attrs_vectorizer)
    failsFile.write(fails)
    failsFile.close()
    print('-' * 60, file=f)
    print('Results of testing:', file=f)
    predicted_all = []
    for i in range(len(CLASSES_LIST)):
        predicted = model[i].predict(X_test)
        predicted_all.append(predicted)
        print(CLASSES_LIST[i] + ':', file=f)
        print(metrics.classification_report(y_test[i], predicted), file=f)
    print('Full result:', file=f)
    print(metrics.classification_report(np.reshape(y_test, (1, len(y_test)*len(y_test[0])))[0],
                                        np.reshape(np.array(predicted_all), (1, len(y_test) * len(y_test[0])))[0]),
          file=f)
    # print(metrics.confusion_matrix(y_test, predicted))
    f.close()
    print('Done. Results are in results.txt')


if __name__ == '__main__':
    train_and_test()
