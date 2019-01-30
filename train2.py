import pandas as pd
import numpy as np
from datagen import TimeseriesGenerator
from models import stupidcnn
from keras import optimizers as ko, callbacks as kc
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

if __name__ == '__main__':
    csv = pd.read_csv('../input/train.csv',
                      dtype={"acoustic_data": np.int8, "time_to_failure": np.float32}, )
    acoustic_data = np.expand_dims(np.array(csv['acoustic_data']), 1)
    time_to_failure = np.expand_dims(np.array(csv['time_to_failure']), 1)

    nb_train = int(len(acoustic_data) * .8)
    train_acoustic_data = acoustic_data[:nb_train]
    train_time_to_failure = time_to_failure[:nb_train]
    train_generator = TimeseriesGenerator(train_acoustic_data,
                                          train_time_to_failure[:, 0], 1024, batch_size=256, shuffle=True)
    test_acoustic_data = acoustic_data[nb_train:]
    test_time_to_failure = time_to_failure[nb_train:]
    test_generator = TimeseriesGenerator(test_acoustic_data,
                                         test_time_to_failure[:, 0], 1024, batch_size=256)

    model = stupidcnn()
    model.compile(ko.SGD(0.01, 0.9), 'mse')
    model.load_weights('weights/morestupidcnn.h5')

    callback = kc.ModelCheckpoint('weights/stupidcnn.h5',
                                  monitor='val_acc',
                                  mode='max',
                                  save_best_only=True)

    model.fit_generator(train_generator,
                        epochs=20,
                        steps_per_epoch=len(train_generator)//10,
                        validation_data=test_generator,
                        validation_steps=len(test_generator)//10
                        )
    model.save_weights('weights/stupidcnn-temp.h5')
