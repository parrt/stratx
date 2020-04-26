from support import *
from sklearn.datasets import load_boston

np.random.seed(1)

# use just one trial for ~500 records otherwise bootstrapping uses too little data
gen_topk_figs(n_trials=1,dataset="boston",targetname='MEDV',
              title="Boston Housing Prices",
              min_slopes_per_x=1,                 # too little data; don't filter out anything
              yrange=(0, 5),
              yunits="k$")