import json

from random import shuffle
import os
import numpy as np

from lxml import etree as ET
from lxml.etree import HTMLParser
from scipy.sparse import vstack, hstack


def sign(x):
    return 1 - (x < 0)


def get_site_classes(classes):
    sites_with_classes = dict()
    with open('classes_list.txt', 'w') as thefile:
        for key, val in classes.items():
            for site in val:
                if site.startswith('www.'):
                    site = site[4:]
                sites_with_classes[site] = key
                thefile.write("{0}: {1}\n".format(site, key))
    return sites_with_classes


WEBPAGES_PATH = 'webpages/'
RESULTS_PATH = 'results/'
if not os.path.isdir(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
VECTORIZER_DATA_FILE = ['text_vect_data.txt', 'attrs_vect_data.txt']
PAGE_CLASSES_FILE = './metadata.json'
PAGE_CLASSES = dict()
CLASSES_LIST = []

with open(PAGE_CLASSES_FILE) as data:
    PAGE_CLASSES = json.load(data)
    CLASSES_LIST = list(PAGE_CLASSES.keys())


SITES_WITH_CLASSES = get_site_classes(PAGE_CLASSES)


def load_sites(path):
    sites_list = []

    for dirname in os.listdir(path):
        dir = os.path.join(path, dirname)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            sites_list.append((file_path, dirname, dir))

    shuffle(sites_list)

    split_line = int(len(sites_list)/4*3)
    return sites_list[:split_line], sites_list[split_line:]


def get_class_index(site_name):
    return CLASSES_LIST.index(SITES_WITH_CLASSES[site_name])


def load_file(file):
    # print(file)

    tree = ET.parse(file, parser=HTMLParser())
    if tree.getroot() is None:
        raise Exception('Can\'t get root.')

    title = tree.getroot().find('title')
    if title is None:
        title = ''
    body = tree.getroot().find('body')
    if body is None:
        raise Exception('Can\'t find the BODY tag.')

    text_train_data = [title]
    attr_train_data = []

    for node in body.iter():
        if node.text:
            text_train_data.append(node.text)
        # attr_vals = []
        # for attr in node.items():
        #     if attr[1]:
        #         attr_vals.append(attr[1])
        if node.get('class'):
            attr_train_data.append(node.get('class'))

    if not len(text_train_data) or not len(attr_train_data):
        raise Exception('Empty data for learning.')

    return ' \n '.join(text_train_data), ' \n '.join(np.array(attr_train_data))


def corpus_transformation(sites, text_vectorizer, attrs_vectorizer):
    y = []
    X = None
    fails = ''
    for site in sites:
        try:
            text, attrs = load_file(site[0])
        except Exception as exception:
            fails += '{0}: {1}\n'.format(site[1], exception)
        if not isinstance(text, str):
            print('text is', type(text))
        y.append(get_class_index(site[1]))
        X_row = hstack([
            text_vectorizer.transform([text]),
            # Uncomment this string to add attributes data into learning data
            # attrs_vectorizer.transform([attrs])
        ])
        if X is None:
            X = X_row
        else:
            X = vstack((X, X_row))
    # print(type(y))
    # print(len(y))
    # print(X.shape)
    answers_formatted = []
    answers = y
    for i in range(len(CLASSES_LIST)):
        answers_formatted.append([sign(x-i)*sign(i-x) for x in answers])
    return X, np.array(answers_formatted), fails