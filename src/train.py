import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .model import build_eegnet


def load_preprocessed(save_path: str):
    X_list, y_list = [], []

    files = sorted([f for f in os.listdir(save_path) if f.endswith("_X.npy")])

    for f in files:
        X = np.load(os.path.join(save_path, f))
        y = np.load(os.path.join(save_path, f.replace("_X.npy", "_y.npy")))
        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)

    X_all = X_all[..., np.newaxis]
    return X_all, y_all


def train_model(save_path: str, model_save_path: str):

    X_all, y_all = load_preprocessed(save_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = build_eegnet(
        n_channels=X_all.shape[1],
        n_times=X_all.shape[2],
        dropout_rate=0.3
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=30,
        validation_data=(X_val, y_val),
        shuffle=True
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    return model, history, X_test, y_test