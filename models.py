from keras import layers as kl, models as km


def stupidlstm():
    inp = kl.Input((512, 1))
    x = kl.GRU(2048,)(inp)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)


def stupidgrus():
    inp = kl.Input((1024, 1))
    x = kl.GRU(256, return_sequences=True)(inp)
    x = kl.GRU(256,)(x)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)


def stupidcnn():
    inp = kl.Input((1024, 1))
    x = kl.Conv1D(128, 3)(inp)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.Conv1D(128, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(256, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.Conv1D(256, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(512, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.Conv1D(512, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)
