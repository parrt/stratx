from sklearn.datasets import load_boston, load_diabetes
from stratx.partdep import *
from stratx.ice import *


def original_pdp(model, X, colname, targetname):
    ice = predict_ice(model, X, colname, targetname)
    #  Row 0 is actually the sorted unique X[colname] values used to get predictions.
    avg_y = np.mean(ice[1:], axis=0)
    min_pdp_y = avg_y[0]
    # if 0 is in x feature and not on left/right edge, get y at 0
    # and shift so that is x,y 0 point.
    linex = ice.iloc[0, :]  # get unique x values from first row
    nx = len(linex)
    if linex[int(nx * 0.05)] < 0 or linex[-int(nx * 0.05)] > 0:
        closest_x_to_0 = np.argmin(
            np.abs(np.array(linex - 0.0)))  # do argmin w/o depr warning
        min_pdp_y = avg_y[closest_x_to_0]

    avg_y -= min_pdp_y
    return avg_y.values



boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
X = df.drop('MEDV', axis=1)
y = df['MEDV']
#print(df.columns)

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)

pdp = original_pdp(rf, X, 'LSTAT', 'MEDV')
print(pdp)
