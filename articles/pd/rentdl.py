import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(1)  # pick seed for reproducible article images

from keras import models, layers, callbacks, optimizers

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

model = models.Sequential()
layer1 = 3000
layer2 = 3000
layer3 = 3000
layer4 = 3000
layer5 = 1000
batch_size = 200
model.add(layers.Dense(layer1, input_dim=X.shape[1], activation='relu'))
model.add(layers.Dense(layer2, activation='relu'))
model.add(layers.Dense(layer3, activation='relu'))
model.add(layers.Dense(layer4, activation='relu'))
model.add(layers.Dense(layer5, activation='relu'))
model.add(layers.Dense(1))

opt = optimizers.Adam()  # lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mae'])

callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X, y,
                    epochs=1000,
                    validation_split=0.2,
                    batch_size=batch_size,
                    # callbacks=[callback]
                    )

y_pred = model.predict(X)
# y_pred *= np.std(y_raw)  # undo normalization on y
# y_pred += np.mean(y_raw)
r2 = r2_score(y_raw, y_pred)
print("Keras training R^2", r2)
