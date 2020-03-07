''''
  data_loader.py
  Genode deep leaning
'''

import json
import tarfile

import numpy as np

from GUtils.data_utils import get_file

"""  """


def convert_tsv_to_csv(tsv_file, csv_file=''):
    with open(file=tsv_file)as fin, open(csv_file, 'w') as fout:
        for line in fin:
            fout.write(line.replace('\t', ','))

def convert_tab_to_csv(tab_file, csv_file=''):
    with open(file=tab_file)as fin, open(csv_file, 'w') as fout:
        print("Opening file",tab_file)
        for line in fin:
            print("Iterating and writing file")
            fout.write(line.replace('|', ','))


def convert_csv_to_tvs(csv_file, tsv_file=''):
    with open(file=csv_file) as fin, open(tsv_file, 'w')as fout:
        for line in fin:
            fout.write(line.replace(',', '\t'))


def fetch_data(path='database.tar.gz', url=''):
    path = get_file(path, origin=url)
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        file = tar.extractfile(member=member)
        if file is None:
            continue
            file_content = file.read()
            dataset = json.loads(file_content.decode('utf-8'))
            break

        data, label = dataset['data'], dataset['label']
        X = data
        Y = label
        file.close()
        X, Y, = map(lambda element: np.array(list(element)), X), map(lambda element: np.array(element), Y)
        print("X shape ", X.shape, " Y shape ", Y.shape)
        return list(X), list(Y)


def load_data(path):
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        file = tar.extractfile(member=member)
        if file is None:
            continue
            file_content = file.read()
            dataset = json.loads(file_content.decode('utf-8'))
            break
    X, Y = dataset['X'], dataset['Y']
    file.close()
    X, Y = map(lambda element: np.array(list(element)), X), map(lambda element: np.array(element), Y)
    return list(X), list(Y)


def load_json_dataset(filename):
    with open(file=filename) as fout:
        file = fout.read()
        dataset = json.load(file.d)
        X, Y = dataset['X'], dataset['Y']
        fout.close()
        X = map(lambda element: np.array(list(element)), X)
        Y = map(lambda element: np.array(list(element)), Y)
    return list(X), list(Y)


def shuffle_and_split(X, Y, weights, seed=123456, fraction=0.8):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == weights.shape[0]

    # X = X.reshape(X.shape + (1,))
    # weights = weights.reshape(weights.shape + (1,))

    N = X.shape[0]
    print("X shape == ", X.shape, " Y shape == ", Y.shape)
    print("X size == ", X.size, " Y size == ", Y.size)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    idx = int(N * fraction)
    training_idx, test_idx = indices[:idx], indices[idx:]

    (x_train, y_train, weights_train) = (X[training_idx], Y[training_idx], weights[training_idx])
    (x_test, y_test, weights_test) = (X[test_idx], Y[test_idx], weights[test_idx])

    return (x_train, y_train, weights_train), (x_test, y_test, weights_test)


def load_data2(path='peptone_dspp.tar.gz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path, origin='https://peptone.io/dspp/download/database.json.tar.gz?raw=true')
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is None: continue
        content = f.read()
        # print("Load json {}".format(f))
        database = json.loads(content.decode('utf-8'))
        break

    X, Y = database['X'], database['Y']
    f.close()
    X, Y = map(lambda element: np.array(list(element)), X), map(lambda element: np.array(element), Y)
    return list(X), list(Y)


if __name__ == '__main__':
    print("Hello Python!")
