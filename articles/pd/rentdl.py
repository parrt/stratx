# On linux box: source activate tensorflow2_p36
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

from keras import models, layers, callbacks, optimizers, regularizers

np.random.seed(1)  # pick seed for reproducible article images


def load_rent(n:int=None, clean_prices=True):
    """
    Download train.json from https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
    and save into data subdir.
    """
    # df = pd.read_json(f'train.json')
    df = pd.read_csv('rent10k.csv')
    print(f"Rent has {len(df)} records")

    X = df.drop('price', axis=1)
    y = df['price']
    return X, y


X_raw, y_raw = load_rent(n=10_000)

X, y = X_raw.copy(), y_raw.copy()

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# y = (y - np.mean(y))/np.std(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = models.Sequential()
layer1 = 1400
layer2 = 1400
layer3 = 1400
batch_size = 2000
dropout = 0.4
model.add(layers.Dense(layer1, input_dim=X.shape[1], activation='relu')
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout))

model.add(layers.Dense(layer2, activation='relu', activity_regularizer)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout))

model.add(layers.Dense(layer3, activation='relu', activity_regularizer)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout))

model.add(layers.Dense(1))

#learning_rate=1e-2 #DEFAULT
# SGB gets NaNs?
#opt = optimizers.SGD(lr=0.01)
opt = optimizers.Adam(lr=0.1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mae'])

callback = callbacks.EarlyStopping(monitor='val_loss', patience=40)
history = model.fit(X_train, y_train,
                    #epochs=1000,
                    epochs=300,
#                    validation_split=0.2,
                    validation_data=(X_test, y_test),
                    batch_size=batch_size,
                    #callbacks=[callback],
                    verbose=1
                    )

y_pred = model.predict(X_train)
# y_pred *= np.std(y_raw)  # undo normalization on y
# y_pred += np.mean(y_raw)
r2 = r2_score(y_train, y_pred)
print("Keras training R^2", r2)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Keras validation R^2", r2)
