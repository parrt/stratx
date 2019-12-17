import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from timeit import default_timer as timer

import shap

df = pd.read_feather("data/league.feather")
X = df.drop('win', axis=1)
y = df['win']

# create train/validation split
Xt, Xv, yt, yv = train_test_split(X,y, test_size=0.2)

ntrees = 50
rf = RandomForestClassifier(n_estimators=ntrees, n_jobs=-1)
start = timer()
rf.fit(Xt, yt)
stop = timer()
print(f"RF fit time with {len(Xt)} records and {ntrees} trees = {(stop-start):.2f}s")

"""
OUTPUT:
RF fit time with 1229705 records and 50 trees = 128.03s
TreeExplainer(rf, feature_perturbation="tree_path_dependent") time on RF with 100 test records = 155.38s
TreeExplainer(rf, feature_perturbation="tree_path_dependent") time on RF with 500 test records = 788.71s
TreeExplainer(rf, feature_perturbation="tree_path_dependent") time on RF with 1000 test records = 1595.54s

So about 1.5s per record
"""
# for size in [100, 500, 1000]:
#     Xv_ = Xv[:size]
#     start = timer()
#     explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
#     shap_values = explainer.shap_values(Xv_)
#     stop = timer()
#     print(f'''TreeExplainer(rf, feature_perturbation="tree_path_dependent") time on RF with {len(Xv_)} test records = {(stop-start):.2f}s''')


# Xt_ = Xt[:100]
# Xv_ = Xv[:100]
# start = timer()
# explainer = shap.TreeExplainer(rf, Xt_, feature_perturbation="interventional")
# shap_values = explainer.shap_values(Xv_)
# stop = timer()
# print(f'''TreeExplainer(rf, feature_perturbation="interventional") time on RF with {len(Xv_)} test records = {(stop-start):.2f}s''')

# Old crash
start = timer()
explainer = shap.TreeExplainer(rf, data=shap.sample(Xt, 100), feature_perturbation='interventional')
shap_values = explainer.shap_values(Xv[:100])
stop = timer()
print(f"SHAP time {(stop - start):.1f}s")
print(f"SHAP time for {len(Xv)} records = {(stop - start):.1f}s")

