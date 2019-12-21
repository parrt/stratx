from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import shap

import numpy as np
import pandas as pd

def synthetic_poly_dup_data(n):
    p = 3 # x1, x2, x3
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 10
    # copy x1 into x3 with zero-centered noise
    df['x3'] = df['x1'] + np.random.random_sample(size=n)-0.5
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_3 = x_1 + noise, x_1, x_2 ~ U(0,10)"
    return df, eqn

df, eqn = synthetic_poly_dup_data(1000)
X = df.drop('y', axis=1)
y = df['y']

print(eqn)

# Use OLS to get coeff and then LinearExplainer

lm = LinearRegression()
lm.fit(X,y)
print("OLS coeff", lm.coef_)
y_pred = lm.predict(X)
print(f"OLS Training MSE {np.mean((y - y_pred) ** 2):.5f}")

explainer = shap.LinearExplainer(lm, X, feature_dependence='independent')
shap_values = explainer.shap_values(X)
shapimp = np.mean(np.abs(shap_values), axis=0)
print("OLS SHAP importances", shapimp)

beta_stderr = sm.OLS(y, X).fit().bse
print(f"linear model coefficient stderr {beta_stderr.values}\n(high variance on x1 and x3 as expected)")

# Try now with RF model and TreeExplainer

for i in range(5): # try 3 times to see variance in x_1, x_3
    print(f"Trial {i+1}")
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(X, y)
    print(f"RF Training MSE {np.mean((y - rf.predict(X)) ** 2):.5f}")
    explainer = shap.TreeExplainer(rf, data=X, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    print("RF SHAP importances", shapimp)