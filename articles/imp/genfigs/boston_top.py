from support import *
from sklearn.datasets import load_boston

np.random.seed(1)

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# don't need many trials for 500 records
gen_topk_figs(X,y,kfolds=5,n_trials=1,dataset="boston",title="Boston Housing Prices",
              min_samples_leaf = 3, # too little data so need tight groups/leaves
              min_slopes_per_x=1, # too little data
              yrange=(2, 6.5), yunits="k$")