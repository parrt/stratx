# Code to gen .csv files for use with R

from support import synthetic_interaction_data, load_flights

df = synthetic_interaction_data(n=1000)
df.to_csv("interaction.csv", index=False)

X, y, df = load_flights(n=20_000)
df.to_csv("flights20k.csv", index=False)
