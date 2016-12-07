import os


def write_to_file(dir, filename, data):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename, 'w') as file:
        file.write(data)


def shuffle_set(set, mix):
    response = list(range(len(mix)))
    for i in list(range(len(mix))):
        response[mix[i]] = set[i]
    return response


class LogFile:
    def __init__(self, file):
        self.file = open(file, 'rw')

    def write(self, site, exception):
        self.file.write()