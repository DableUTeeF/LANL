import pandas as pd
import numpy as np
from datagen import TimeseriesGenerator
from models import abitdeepgrus
from keras import optimizers as ko
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

if __name__ == '__main__':
    csv = pd.read_csv('misc/train.csv',
                      dtype={"acoustic_data": np.int8, "time_to_failure": np.float32}, )
    acoustic_data = np.expand_dims(np.array(csv['acoustic_data']), 1)
    time_to_failure = np.expand_dims(np.array(csv['time_to_failure']), 1)

    nb_train = int(len(acoustic_data) * .8)
    train_acoustic_data = acoustic_data[:nb_train]
    train_time_to_failure = time_to_failure[:nb_train]
    train_generator = TimeseriesGenerator(train_acoustic_data,
                                          train_time_to_failure[:, 0], 1024, batch_size=256,
                                          shuffle=True)
    test_acoustic_data = acoustic_data[nb_train:]
    test_time_to_failure = time_to_failure[nb_train:]
    test_generator = TimeseriesGenerator(test_acoustic_data,
                                         test_time_to_failure[:, 0], 1024, batch_size=256)

    model = abitdeepgrus()
    model.compile(ko.SGD(0.001, 0.9), 'mse')

    model.fit_generator(train_generator,
                        epochs=10,
                        steps_per_epoch=10000,
                        validation_data=test_generator,
                        validation_steps=2000)
    model.save_weights('weights/stupidgru.h5')
