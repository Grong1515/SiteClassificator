import codecs
import os


def write_to_file(dir, filename, data):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with codecs.open(dir + filename, mode='w', encoding='utf-8', errors='ignore') as file:
        file.write(data)


def shuffle_set(set, mix):
    response = list(range(len(mix)))
    for i in list(range(len(mix))):
        response[mix[i]] = set[i]
    return response


class LogFile:
    def __init__(self, file):
        self.file = codecs.open(file, mode='rw', encoding='utf-8', errors='ignore')

    def __del__(self):
        self.file.close()

    def write(self, site, exception):
        self.file.write()