# Code to gen .csv files for use with R

from articles.pd.support import *

X, y = load_bulldozer(n=10_000)
df = pd.concat([X, y], axis=1)
df.to_csv("bulldozer10k.csv", index=False)

X, y = load_rent(n=10_000)
df = pd.concat([X, y], axis=1)
df.to_csv("rent10k.csv", index=False)

df = toy_weather_data()
df.sort_values('state').to_csv("weather.csv", index=False)

X, y, df, eqn = toy_weight_data(2000)
df.to_csv("weight.csv", index=False)

df = synthetic_interaction_data(n=2000)
df.to_csv("interaction.csv", index=False)

