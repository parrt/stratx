from stratx.ice import *
from stratx.support import *

np.random.seed(1)  # pick seed for reproducible article images

from keras import models, layers, callbacks, optimizers

def load_rent(n:int=None, clean_prices=True):
    """
    Download train.json from https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
    and save into data subdir.
    """
    df = pd.read_json(f'{datadir}/train.json')
    print(f"Rent has {len(df)} records")

    # Create ideal numeric data set w/o outliers etc...

    if clean_prices:
        df = df[(df.price > 1_000) & (df.price < 10_000)]

    df = df[df.bathrooms <= 6]  # There's almost no data for 6 and above with small sample
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df['interest_level'] = df['interest_level'].map({'low': 1, 'medium': 2, 'high': 3})
    df["num_desc_words"] = df["description"].apply(lambda x: len(x.split()))
    df["num_features"] = df["features"].apply(lambda x: len(x))
    df["num_photos"] = df["photos"].apply(lambda x: len(x))

    hoods = {
        "hells": [40.7622, -73.9924],
        "astoria": [40.7796684, -73.9215888],
        "Evillage": [40.723163774, -73.984829394],
        "Wvillage": [40.73578, -74.00357],
        "LowerEast": [40.715033, -73.9842724],
        "UpperEast": [40.768163594, -73.959329496],
        "ParkSlope": [40.672404, -73.977063],
        "Prospect Park": [40.93704, -74.17431],
        "Crown Heights": [40.657830702, -73.940162906],
        "financial": [40.703830518, -74.005666644],
        "brooklynheights": [40.7022621909, -73.9871760513],
        "gowanus": [40.673, -73.997]
    }
    for hood, loc in hoods.items():
        # compute manhattan distance
        df[hood] = np.abs(df.latitude - loc[0]) + np.abs(df.longitude - loc[1])
        df[hood] *= 1000 # GPS range is very tight so distances are very small. bump up
    hoodfeatures = list(hoods.keys())

    if n is not None:
        howmany = min(n, len(df))
        df = df.sort_values(by='created').sample(howmany, replace=False)
    # df = df.sort_values(by='created')  # time-sensitive dataset
    # df = df.iloc[-n:]

    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price',
                  'interest_level']+
                 hoodfeatures+
                 ['num_photos', 'num_desc_words', 'num_features']]
    # print(df_rent.head(3))

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    return X, y


X_raw, y_raw = load_rent(n=10_000)

X, y = X_raw.copy(), y_raw.copy()

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# y = (y - np.mean(y))/np.std(y)

model = models.Sequential()
layer1 = 500
layer2 = 500
layer3 = 500
layer4 = 500
layer5 = 256
batch_size = 200
model.add(layers.Dense(layer1, input_dim=X.shape[1], activation='relu'))
model.add(layers.Dense(layer2, activation='relu'))
model.add(layers.Dense(layer3, activation='relu'))
# model.add(layers.Dense(layer4, activation='relu'))
# model.add(layers.Dense(layer5, activation='relu'))
model.add(layers.Dense(1))

opt = optimizers.Adam()  # lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
# model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mae'])

callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X, y,
                    epochs=3000,
                    validation_split=0.2,
                    batch_size=batch_size,
                    # callbacks=[callback]
                    )

y_pred = model.predict(X)
# y_pred *= np.std(y_raw)  # undo normalization on y
# y_pred += np.mean(y_raw)
r2 = r2_score(y_raw, y_pred)
print("Keras training R^2", r2)
plt.ylabel("MAE")
plt.xlabel("epochs")

plt.plot(history.history['val_mae'], label='val_mae')
plt.plot(history.history['mae'], label='train_mae')
plt.title(f"batch_size {batch_size}, Layer1 {layer1}, Layer2 {layer2}, R^2 {r2:.3f}")
plt.legend()
plt.show()
