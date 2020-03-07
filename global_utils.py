"""Python utilities required Keras."""
from __future__ import absolute_import

import numpy as np
import random
import pandas as pd

import time
import sys
import six
import marshal
import types as python_types
import inspect
import tensorflow as tf
from keras.callbacks import Callback

_GLOBAL_CUSTOM_OBJECTS = {}


class CustomObjectScope(object):
    """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

    Code within a `with` statement will be able to access custom objects
    by name. Changes to global custom objects persist
    within the enclosing `with` statement. At end of the `with` statement,
    global custom objects are reverted to state
    at beginning of the `with` statement.

    # Example

    Consider a custom object `MyObject`

    ```python
        with CustomObjectScope({'MyObject':MyObject}):
            layer = Dense(..., kernel_regularizer='MyObject')
            # save, load, etc. will recognize custom object by name
    ```
    """

    def __init__(self, *args):
        self.custom_objects = args
        self.backup = None

    def __enter__(self):
        self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
        for objects in self.custom_objects:
            _GLOBAL_CUSTOM_OBJECTS.update(objects)
        return self

    def __exit__(self, *args, **kwargs):
        _GLOBAL_CUSTOM_OBJECTS.clear()
        _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


def custom_object_scope(*args):
    """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

    Convenience wrapper for `CustomObjectScope`.
    Code within a `with` statement will be able to access custom objects
    by name. Changes to global custom objects persist
    within the enclosing `with` statement. At end of the `with` statement,
    global custom objects are reverted to state
    at beginning of the `with` statement.

    # Example

    Consider a custom object `MyObject`

    ```python
        with custom_object_scope({'MyObject':MyObject}):
            layer = Dense(..., kernel_regularizer='MyObject')
            # save, load, etc. will recognize custom object by name
    ```

    # Arguments
        *args: Variable length list of dictionaries of name,
            class pairs to add to custom objects.

    # Returns
        Object of type `CustomObjectScope`.
    """
    return CustomObjectScope(*args)


def get_custom_objects():
    """Retrieves a live reference to the global dictionary of custom objects.

    Updating and clearing custom objects using `custom_object_scope`
    is preferred, but `get_custom_objects` can
    be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

    # Example

    ```python
        get_custom_objects().clear()
        get_custom_objects()['MyObject'] = MyObject
    ```

    # Returns
        Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
    """
    return _GLOBAL_CUSTOM_OBJECTS


def serialize_keras_object(instance):
    if instance is None:
        return None
    if hasattr(instance, 'get_config'):
        return {
            'class_name': instance.__class__.__name__,
            'config': instance.get_config()
        }
    if hasattr(instance, '__name__'):
        return instance.__name__
    else:
        raise ValueError('Cannot serialize', instance)


def deserialize_keras_object(identifier, module_objects=None,
                             custom_objects=None,
                             printable_module_name='object'):
    if isinstance(identifier, dict):
        # In this case we are dealing with a Keras config dictionary.
        config = identifier
        if 'class_name' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))
        class_name = config['class_name']
        if custom_objects and class_name in custom_objects:
            cls = custom_objects[class_name]
        elif class_name in _GLOBAL_CUSTOM_OBJECTS:
            cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
        else:
            module_objects = module_objects or {}
            cls = module_objects.get(class_name)
            if cls is None:
                raise ValueError('Unknown ' + printable_module_name +
                                 ': ' + class_name)
        if hasattr(cls, 'from_config'):
            arg_spec = inspect.getargspec(cls.from_config)
            if 'custom_objects' in arg_spec.args:
                custom_objects = custom_objects or {}
                return cls.from_config(config['config'],
                                       custom_objects=dict(list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                                                           list(custom_objects.items())))
            return cls.from_config(config['config'])
        else:
            # Then `cls` may be a function returning a class.
            # in this case by convention `config` holds
            # the kwargs of the function.
            return cls(**config['config'])
    elif isinstance(identifier, six.string_types):
        function_name = identifier
        if custom_objects and function_name in custom_objects:
            fn = custom_objects.get(function_name)
        elif function_name in _GLOBAL_CUSTOM_OBJECTS:
            fn = _GLOBAL_CUSTOM_OBJECTS[function_name]
        else:
            fn = module_objects.get(function_name)
            if fn is None:
                raise ValueError('Unknown ' + printable_module_name +
                                 ':' + function_name)
        return fn
    else:
        raise ValueError('Could not interpret serialized ' +
                         printable_module_name + ': ' + identifier)


def func_dump(func):
    """Serializes a user defined function.

    # Arguments
        func: the function to serialize.

    # Returns
        A tuple `(code, defaults, closure)`.
    """
    code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
    """Deserializes a user defined function.

    # Arguments
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    # Returns
        A function object.
    """
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
        if isinstance(defaults, list):
            defaults = tuple(defaults)
    code = marshal.loads(code.encode('raw_unicode_escape'))
    if globs is None:
        globs = globals()
    return python_types.FunctionType(code, globs,
                                     name=code.co_name,
                                     argdefs=defaults,
                                     closure=closure)


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class LossRatioLogger(Callback):

    """"Override function from Callback class that logs the loss and val_loss ratio"""
    def on_epoch_end(self, epoch, logs = {}):
        ratio = logs.get('loss')/logs.get('val_loss')
        print("Ratio(l/v_l) = {:2.2f}".format(ratio))

def protein_seq_2oneHot(sequence):
    """
        Return a binary one-hot vector
    """
    one_digit = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, \
        'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, \
        'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    assert len(sequence) >= 1
    encoded = []
    for letter in sequence:
        tmp = np.zeros(20)
        tmp[one_digit[letter]] = 1
        encoded.append(tmp)
    assert len(encoded) == len(sequence)
    encoded = np.asarray(encoded)
    return list(encoded.flatten())

def seq(sequence):
    Alph = {'A': 1, 'a': 1,
            'B': 2, 'b': 2,
            'C': 3, 'c': 3,
            'D': 4, 'd': 4,
            'E': 5, 'e': 5,
            'F': 6, 'f': 6,
            'G': 7, 'g': 7,
            'H': 8, 'h': 8,
            'I': 9, 'i': 9,
            'J': 10, 'j': 10,
            'K': 11, 'k': 11,
            'L': 12, 'l': 12,
            'M': 13, 'm': 13,
            'N': 14, 'n': 14,
            'O': 15, 'o': 15,
            'P': 16, 'p': 16,
            'Q': 17, 'q': 17,
            'R': 18, 'r': 18,
            'S': 19, 's': 19,
            'T': 20, 't': 20,
            'U': 21, 'u': 21,
            'V': 22, 'v': 22,
            'W': 23, 'w': 23,
            'X': 24, 'x': 24,
            'Y': 25, 'y': 25,
            'Z': 26, 'z': 26
            }

    dataset = []
    for d in sequence:
        d1 = []
        for letters in d:
            d1.append(np.float32(Alph[letters]))
        for j in range(20 - len(d1)):
            d1.append(np.float32(0))
        dataset.append(d1)
        return list(dataset)

def rmsd(y, prediction):
    """"Compute Root Mean Square Defference"""
    return tf.sqrt(tf.reduce_mean(tf.pow(prediction - y, 2)))

def chi2(exp, obs):
    """
        Compute CHI^2 statistics of non-zero expected elements
    """
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(exp, zero)


    def masking(tensor, mask):
        return tf.boolean_mask(tensor, mask)

    stat = tf.reduce_sum(
        tf.div(
            tf.pow(
                tf.subtract(masking(obs, mask), masking(exp, mask)),
                2),
            masking(exp, mask)),
        name="chi2_statistics")

    return stat

def generate_weights(array):
    results = []
    for row in array:
        weigth = np.zeros(len(row))
        mask = row != 0.0
        weigth[mask] = 1.0
        results.append(weigth)
    weights = np.array(results)
    return weights

def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def split_data_to_train_test(data, label, test_portion):

    dataset = list(zip(data, label))
    # pair corresponding values
    train, test = split_data(data=dataset, prob= 1 - test_portion)  # split the dataset of pairs
    x_train, y_train = list(zip(*train))          # magical un-zip trick
    x_test, y_test = list(zip(*test))
    #convert list to numpy array
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train, x_test, y_train, y_test

def load_data_seq_csv(filepath, x_descrip='', y_descrip=''):
    dataframe = pd.read_csv(filepath)
    print("Data shape: ", dataframe.shape)
    dataset = dataframe
    X, Y = dataset[x_descrip], dataset[y_descrip]
    X, Y = map(lambda element: np.array(list(element)), X), map(lambda element: np.array(element), Y)
    return list(X), list(Y)