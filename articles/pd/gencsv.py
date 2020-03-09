# Code to gen .csv files for use with R

from stratx.support import *

X, y = load_bulldozer(n=10_000)
df = pd.concat([X, y], axis=1)
df.to_csv("bulldozer10k.csv", index=False)

X, y = load_rent(n=10_000)
df = pd.concat([X, y], axis=1)
df.to_csv("rent10k.csv", index=False)

df_yr1 = toy_weather_data()
df_yr1['year'] = 1980
df_yr2 = toy_weather_data()
df_yr2['year'] = 1981
df_yr3 = toy_weather_data()
df_yr3['year'] = 1982
df = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
df.to_csv("weather.csv", index=False)

X, y, df, eqn = toy_weight_data(2000)
df.to_csv("weight.csv", index=False)
