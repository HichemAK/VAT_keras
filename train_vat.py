import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Nombre de classes
from vat_model import VAT

num_classes = 10

# Dimension des données en entrée (image noir et blanc 28x28)
input_shape = (28, 28, 1)

# Chargement du dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# On ne garde que 100 images labellisées tirées aléatoirement de x_train tout en gardant 10 exemples de chaque classe
x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, train_size=100, stratify=y_train, random_state=12375)

# Validation set (on garde 100 exemples de chaque classe pou la validation)
x_ul, x_val, _, y_val = train_test_split(x_temp, y_temp, test_size=1000, stratify=y_temp, random_state=1627)

# On divise les valeurs des pixels par 255 pour les ramener dans l'intervalle [0, 1]
x_train = x_train.astype("float32") / 255
x_val = x_val.astype("float32") / 255
x_ul = x_ul.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# On ajoute une dimension aux données pour qu'elles soient compatibles avec les Convolutional Layers
x_train = np.expand_dims(x_train, -1)
x_ul = np.expand_dims(x_ul, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_val.shape[0], "val samples")
print(x_ul.shape[0], "unlabeled samples")
print(x_test.shape[0], "test samples")


# One-hot encoding des classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Le modèle est une suite de Relu 2D Convolutional Layers 3x3 avec un MaxPooling2D entre chaque couche.
# On ajoute une couche Dense avec dropout + softmax à la fin pour classifier l'image.
model = VAT(eps=2)


# Générateur de données utilisé pour l'entrainement
def data_generator(x_train, x_ul, y_train,
                 batch_size_l, batch_size_ul):
    x_all = np.concatenate([x_train, x_ul])
    while True:
        sample_l = np.random.randint(x_train.shape[0], size=batch_size_l)
        sample_x_l = x_train[sample_l]
        sample_y_l = y_train[sample_l]

        sample_ul = np.random.randint(x_all.shape[0], size=batch_size_ul)
        sample_x_ul = x_all[sample_ul]
        yield (sample_x_l, sample_x_ul), (sample_y_l, )


# Batch size utilisé pour calculer l'erreur de classification et l'erreur R_vadv respectivement
batch_size_l = 64
batch_size_ul = 256
epochs = 300

# On implémente un Early Stopping avec patience = 15
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, mode='max', restore_best_weights=True)

# Les paramètres par défaut de Adam ont été utilisés
adam = keras.optimizers.Adam(learning_rate=10**-4)
model.compile(optimizer=adam)

# Début de l'entrainement
h = model.fit(data_generator(x_train, x_ul, y_train, batch_size_l, batch_size_ul), validation_data=(x_val, y_val),
              callbacks=[callback], epochs=epochs, steps_per_epoch=200)

# Evaluation sur le Test Set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# NOTES:
# Test loss: 0.5341874957084656
# Test accuracy: 0.8622000217437744
