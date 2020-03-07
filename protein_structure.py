''''
 model.py
    Deep learning Protein model
'''
from keras.models import Sequential, Input, Model
from keras.layers import Dense,Dropout, Convolution2D, Convolution1D, Flatten, MaxPooling2D, MaxPooling1D
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

from keras.losses import logcosh
from keras.preprocessing.sequence import pad_sequences
from GUtils.data_loader import shuffle_and_split, load_data, fetch_data, load_data2, load_json_dataset
from GUtils.global_utils import Struct, protein_seq_2oneHot, generate_weights, LossRatioLogger, rmsd, chi2
import keras

# all free parameters for the modtarfile.ReadError: not a gzip fileel
parameters = {"maxlen": 16000,
              "output": 800,
              "hidden_1": 200,
              "hidden_2": 400}

X, Y = load_data2()
    #fetch_data(url='https://peptone.io/dspp/download/database.json.tar.gz?raw=true')

weights = generate_weights(Y)
X = [protein_seq_2oneHot(x) for x in X]
X = pad_sequences(X, maxlen=parameters['maxlen'])
Y = pad_sequences(Y, maxlen=parameters['output'], dtype='float32')
weights = pad_sequences(weights, parameters['output'], dtype='float32')


def deep_protein_dense():
    inputs = Input(shape=(parameters['maxlen'],))
    net = Dense(units=parameters['hidden_1'], activation='relu')(inputs)
    net = Dense(units=parameters['hidden_2'], activation='relu')(net)
    net = Dense(units=parameters['output'], activation='softmax')(net)
    pred = Model(inputs=inputs, outputs=net)
    return pred

def deep_protein_cnn():
    protein_input = Input(shape=(1600, 800))
    conv1 = Convolution1D(filters=800, kernel_size=2, padding='same', activation='relu')(protein_input)
    conv2 = Convolution1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=4, strides=1, padding='same')(conv2)
    conv_flat = Flatten()(pool1)
    net = Dense(units=parameters['hidden_1'], activation='relu')(conv_flat)
    net = Dense(units=parameters['output'], activation='softmax')(net)
    pred = Model(inputs= protein_input, outputs=net)
    return pred




if __name__ == '__main__':
    # Shuffle and split the data
    (x_train, y_train, weights_train), (x_test, y_test, weights_test) = shuffle_and_split(X, Y, weights)
    model = deep_protein_dense()
    #model = deep_protein_cnn()
    model.compile(optimizer=keras.optimizers.Adam(), loss=logcosh, metrics=[rmsd, chi2])

    history = model.fit(x=x_train, y=y_train, epochs=10, batch_size=20,
              validation_data=(x_test, y_test), callbacks=[LossRatioLogger()])
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # evaluate t
    score = model.evaluate(x_test, y_test)

    # Just some simple diagnostics
    print('Test rmsd:', score[0])
    print('Test chi2:', score[1] * 100)

    # Serialize model to JSON
    with open("model.json", "w") as fp:
        fp.write(model.to_json())

    # Serialize weights to HDF5
    model.save_weights("model.h5")
