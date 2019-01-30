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


def abitdeepgrus():
    inp = kl.Input((1024, 1))
    x = kl.GRU(128, return_sequences=True)(inp)
    x = kl.GRU(128, return_sequences=True)(x)
    x = kl.GRU(128, return_sequences=True)(x)
    x = kl.GRU(128, return_sequences=True)(x)
    x = kl.GRU(128, return_sequences=True)(x)
    x = kl.GRU(128, return_sequences=True)(x)
    x = kl.GRU(128,)(x)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)


def stupidcnn():
    inp = kl.Input((1024, 1))
    x = kl.Conv1D(128, 3)(inp)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(256, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(512, 3)(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)


def littelsmartblock(inp, k):
    x = kl.Conv1D(k, 3, padding='same')(inp)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = kl.Conv1D(k, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    return kl.add([inp, x])


def littlesmartcnn():
    inp = kl.Input((1024, 1))
    x = kl.Conv1D(128, 7)(inp)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = littelsmartblock(x, 128)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(256, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = littelsmartblock(x, 256)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(512, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = littelsmartblock(x, 512)
    x = littelsmartblock(x, 512)
    x = littelsmartblock(x, 512)
    x = kl.MaxPooling1D()(x)
    x = kl.Conv1D(1024, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.LeakyReLU()(x)
    x = littelsmartblock(x, 1024)
    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Dense(1)(x)
    return km.Model(inp, x)
