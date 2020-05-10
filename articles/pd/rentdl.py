import numpy as np
import pandas as pd
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import models, layers, callbacks, optimizers
import matplotlib.pyplot as plt
import os


def deeplearning(X_train, X_test, y_train, y_test):
    model = models.Sequential()
    layer1 = 100
    batch_size = 1000
    dropout = 0.3
    model.add(layers.Dense(layer1, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))

    # learning_rate=1e-2 #DEFAULT
    opt = optimizers.SGD()  # SGB gets NaNs?
    # opt = optimizers.RMSprop(lr=0.1)
    opt = optimizers.Adam(lr=0.3)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

    callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train,
                        # epochs=1000,
                        epochs=500,
                        # validation_split=0.2,
                        validation_data=(X_test, y_test),
                        batch_size=batch_size,
                        # callbacks=[tensorboard_callback],
                        verbose=1
                        )

    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    print("Keras training R^2", r2)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Keras validation R^2", r2)

    if True: # Show training result
        y_pred = model.predict(X_test)
        plt.ylabel("MAE")
        plt.xlabel("epochs")

        plt.plot(history.history['val_mae'], label='val_mae')
        plt.plot(history.history['mae'], label='train_mae')
        plt.title(f"batch_size {batch_size}, Layer1 {layer1}")
        plt.legend()
        plt.show()


def rfmodel(X_train, X_test, y_train, y_test):
	rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=.3,
				   oob_score=True, n_jobs=-1)

	rf.fit(X_train, y_train) # Use training set for plotting
	print("RF OOB R^2", rf.oob_score_)
	rf_score = rf.score(X_test, y_test)
	print("RF validation R^2", rf_score)


	# Normalize data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

df = pd.read_csv("rent10k.csv")
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfmodel(X_train, X_test, y_train, y_test)
deeplearning(X_train, X_test, y_train, y_test)
