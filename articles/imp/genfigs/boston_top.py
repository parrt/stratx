from support import *

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

gen_topk_figs(X,y,kfolds=5,n_trials=5,dataset="boston",title="Boston Housing Prices",
              yrange=(2, 6),yunits="k$")