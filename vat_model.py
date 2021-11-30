from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class VAT(keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, alpha=1, eps=10 ** -6):
        super(VAT, self).__init__()

        # L'architecture du modèle
        self.model = keras.Sequential(
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
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        self.kl = tf.keras.losses.KLDivergence()
        self.image_shape = (28, 28, 1)
        self.flatten = layers.Flatten()
        self.eps = eps
        self.alpha = alpha
        self.xi = 10**-6

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.Accuracy(name="accuracy")

    def train_step(self, data):
        # Récupérer les données
        # x_l : Les images labellisées | x_ul : Des images non-labellisées | y_l : les labels de x_l
        (x_l, x_ul), (y_l,) = data

        with tf.GradientTape() as tape:
            # ---- Calcul de la loss de classification ----
            # Prédire les labels des images labellisées
            y_l_pred = self.model(x_l)  # Forward pass
            # Calculer la loss de classification (Cross entropy)
            loss = self.cross_entropy(y_l, y_l_pred)

            # ---- Calcul de la loss du VAT ----
            # Générer un vecteur normé aléatoire
            d = tf.random.normal(tf.shape(x_ul))
            d = d / tf.linalg.norm(self.flatten(d))
            # Multiplier par xi = 10^-6
            r = self.xi * d

            # Prédire les labels des images
            y_ul_pred = self.model(x_ul)
            # Prédire les labels des images avec la perturbation 'r'
            y_vadv = self.model(x_ul + r)

            # Calculer la KL divergence entre les labels des images perturbées et les labels des images non-perturbées
            temp = self.kl(y_ul_pred, y_vadv)

            # Se servir du gradient de cette divergence pour calculer la direction adversariale r_vadv
            # (Plus de détails dans le rapport)
            g = tf.gradients(temp, r, stop_gradients=[r])[0]
            r_vadv = self.eps * g / tf.linalg.norm(self.flatten(g))
            r_vadv = tf.stop_gradient(r_vadv)

            # Ajouter la loss VAT (pondérée avec alpha) à la loss de classification
            loss += self.alpha * self.kl(y_vadv, self.model(x_ul + r_vadv))

        # Calculer les gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Mettre à jour les paramètres du modèle
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Calculer l'accuracy et la loss
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(tf.math.argmax(y_l, axis=-1), tf.math.argmax(y_l_pred, axis=-1))
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    def test_step(self, data):
        # Récupérer les données
        # x : Les images labellisées | y_l : les labels de x_l
        x, y = data
        # Prédire les labels de x
        y_pred = self.model(x)

        # Calculer la loss de classification
        loss = self.cross_entropy(y, y_pred)
        self.loss_tracker.update_state(loss)

        # Calculer l'accuracy
        self.accuracy.update_state(tf.math.argmax(y, axis=-1), tf.math.argmax(y_pred, axis=-1))
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy]
