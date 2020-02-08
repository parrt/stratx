from support import *

np.random.seed(1)

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# don't need many trials for 500 records
gen_topk_figs(X,y,kfolds=2,n_trials=1,dataset="boston",title="Boston Housing Prices",
              min_slopes_per_x=1, # too little data
              yrange=(2, 6), yunits="k$")