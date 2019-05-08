from mine.plot import *
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['wine'] = wine.target

X = df.drop('wine', axis=1)
print(X.head(5))
y = df['wine']

fig, axes = plt.subplots(2, 2)

mine_plot(X, y, 'proline', 'wine', ax=axes[0][0])#, yrange=(-12,0))
#mine_plot(X, y, 'height',    'weight', ax=axes[1][0])#, yrange=(0,160))

plt.show()