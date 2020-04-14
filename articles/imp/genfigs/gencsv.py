# Code to gen .csv files for use with R

from support import synthetic_interaction_data

df = synthetic_interaction_data(n=1000)
df.to_csv("interaction.csv", index=False)

