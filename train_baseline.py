import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Nombre de classes
num_classes = 10

# Dimension des données en entrée (image noir et blanc 28x28)
input_shape = (28, 28, 1)

# Chargement du dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# On ne garde que 100 images labellisées tirées aléatoirement de x_train tout en gardant 10 exemples de chaque classe
x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, train_size=100, stratify=y_train, random_state=12375)

# Validation set (on garde 100 exemples de chaque classe pour la validation)
_, x_val, _, y_val = train_test_split(x_temp, y_temp, test_size=1000, stratify=y_temp, random_state=1627)

# On divise les valeurs des pixels par 255 pour les ramener dans l'intervalle [0, 1]
x_train = x_train.astype("float32") / 255
x_val = x_val.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# On ajoute une dimension aux données pour qu'elles soient compatibles avec les Convolutional Layers
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_val.shape[0], "val samples")
print(x_test.shape[0], "test samples")


# One-hot encoding des classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Le modèle est une suite de Relu 2D Convolutional Layers 3x3 avec un MaxPooling2D entre chaque couche.
# On ajoute une couche Dense avec dropout et un softmax à la fin pour classifier l'image.
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 32
epochs = 300

# On implémente un Early Stopping avec patience = 15 (small dataset)
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, mode='max', restore_best_weights=True)

# Les paramètres par défaut de Adam ont été utilisés
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Début de l'entrainement
h = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[callback])

# Evaluation sur le Test Set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# Meilleur résultat obtenu:
# Test loss (Cross Entropy): 0.5446604490280151
# Test accuracy: 0.864300012588501
