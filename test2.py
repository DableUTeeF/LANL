import os
import pandas as pd
from models import stupidcnn
import numpy as np
from datagen import TimeseriesGenerator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


if __name__ == '__main__':
    model = stupidcnn()
    model.load_weights('weights/morestupidcnn.h5')
    output = []
    with open('prd/morestupidcnn.txt', 'w') as wr:
        wr.write('seg_id,time_to_failure\n')
        for seg in os.listdir("../input/test"):
            csv = pd.read_csv('../input/test/'+seg, dtype={"acoustic_data": np.int8},)
            acoustic_data = np.expand_dims(np.array(csv['acoustic_data']), 1)
            test_generator = TimeseriesGenerator(acoustic_data[-1025:],
                                                 acoustic_data[-1025:], 1024,
                                                 batch_size=256
                                                 )
            out = model.predict_generator(test_generator)
            output.append(out[0].tolist())
            wr.write(seg[:-4])
            wr.write(',')
            wr.write(str(out[0].tolist()[0]))
            wr.write('\n')
