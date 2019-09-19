from sklearn.datasets import load_boston, load_diabetes
from stratx.partdep import *

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

X = df.drop('MEDV', axis=1)
y = df['MEDV']

# WORKS ONLY WITH DATAFRAMES AT MOMENT
plot_stratpd(X, y, 'AGE', 'MEDV', yrange=(-10,10))
plt.tight_layout()
plt.show()
