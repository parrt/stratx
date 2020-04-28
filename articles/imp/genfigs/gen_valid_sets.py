# Code to gen .csv files for use with R

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

from support import load_flights, load_bulldozer, load_rent, datadir

np.random.seed(1)
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
df = df.sample(frac=1.0) # shuffle
df_test = df.iloc[-101:]
df = df[0:405]
df.to_csv(f"{datadir}/boston-train.csv", index=False)
df_test.to_csv(f"{datadir}/boston-test.csv", index=False)

np.random.seed(1)
X, y, df = load_flights(n=25_000)
df = df.sample(frac=1.0) # shuffle
df_test = df.iloc[-5000:]
df = df[0:20_000]
df.to_csv(f"{datadir}/flights-train.csv", index=False)
df_test.to_csv(f"{datadir}/flights-test.csv", index=False)

np.random.seed(1)
X, y = load_rent(n=25_000)
df = pd.concat([X, y], axis=1)
df = df.sample(frac=1.0) # shuffle
df_test = df.iloc[-5000:]
df = df[0:20_000]
df.to_csv(f"{datadir}/rent-train.csv", index=False)
df_test.to_csv(f"{datadir}/rent-test.csv", index=False)

np.random.seed(1)
X, y = load_bulldozer(n=25_000)
df = pd.concat([X, y], axis=1)
df = df.sample(frac=1.0) # shuffle
df_test = df.iloc[-5000:]
df = df[0:20_000]
df.to_csv(f"{datadir}/bulldozer-train.csv", index=False)
df_test.to_csv(f"{datadir}/bulldozer-test.csv", index=False)
