# LSTM and Bi-LSTM model implementation

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2


def set_global_seed(seed):
    """Menetapkan seed di semua sumber acak agar hasil NN reprodusibel/stabil."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class LSTMModel:
    """LSTM regresi deret waktu. Mode residual (default): prediksi = anchor
    y_{t-1} + delta yang dipelajari jaringan; keluaran tetap level."""

    def __init__(self, sequence_length=12, n_features=12, units=64, dropout=0.2,
                 learning_rate=0.001, batch_size=16, epochs=100,
                 residual=True, seed=42, l1=0.0, l2=0.0):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.residual = residual
        self.seed = seed
        self.l1 = l1
        self.l2 = l2
        set_global_seed(seed)
        self.model = self._build_model()

    def _build_model(self):
        reg = l1_l2(l1=self.l1, l2=self.l2)
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            KerasLSTM(self.units, return_sequences=True, kernel_regularizer=reg),
            Dropout(self.dropout),
            KerasLSTM(self.units // 2, return_sequences=False, kernel_regularizer=reg),
            Dropout(self.dropout),
            Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            anchor_train=None, anchor_val=None):
        y_fit = np.asarray(y_train, dtype=float)
        val_data = None
        if self.residual and anchor_train is not None:
            y_fit = y_fit - np.asarray(anchor_train, dtype=float)
            if X_val is not None and anchor_val is not None:
                val_data = (X_val, np.asarray(y_val, float) - np.asarray(anchor_val, float))
        elif X_val is not None:
            val_data = (X_val, y_val)
        callbacks = [EarlyStopping(monitor="val_loss" if val_data is not None else "loss",
                                   patience=10, restore_best_weights=True)]
        self.model.fit(
            X_train, y_fit,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        print("LSTM fitted")

    def predict(self, X, anchor=None):
        out = self.model.predict(X, verbose=0).flatten()
        if self.residual and anchor is not None:
            return out + np.asarray(anchor, dtype=float)
        return out

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)


class BiLSTMModel:
    """BiLSTM untuk regresi deret waktu; lihat LSTMModel untuk mode residual."""

    def __init__(self, sequence_length=12, n_features=12, units=64, dropout=0.2,
                 learning_rate=0.001, batch_size=16, epochs=100,
                 residual=True, seed=42, l1=0.0, l2=0.0):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.residual = residual
        self.seed = seed
        self.l1 = l1
        self.l2 = l2
        set_global_seed(seed)
        self.model = self._build_model()

    def _build_model(self):
        reg = l1_l2(l1=self.l1, l2=self.l2)
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            Bidirectional(KerasLSTM(self.units, return_sequences=True, kernel_regularizer=reg)),
            Dropout(self.dropout),
            Bidirectional(KerasLSTM(self.units // 2, return_sequences=False, kernel_regularizer=reg)),
            Dropout(self.dropout),
            Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            anchor_train=None, anchor_val=None):
        y_fit = np.asarray(y_train, dtype=float)
        val_data = None
        if self.residual and anchor_train is not None:
            y_fit = y_fit - np.asarray(anchor_train, dtype=float)
            if X_val is not None and anchor_val is not None:
                val_data = (X_val, np.asarray(y_val, float) - np.asarray(anchor_val, float))
        elif X_val is not None:
            val_data = (X_val, y_val)
        callbacks = [EarlyStopping(monitor="val_loss" if val_data is not None else "loss",
                                   patience=10, restore_best_weights=True)]
        self.model.fit(
            X_train, y_fit,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        print("BiLSTM fitted")

    def predict(self, X, anchor=None):
        out = self.model.predict(X, verbose=0).flatten()
        if self.residual and anchor is not None:
            return out + np.asarray(anchor, dtype=float)
        return out

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)