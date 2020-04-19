from support import *
import numpy as np

np.random.seed(1)

n = 25_000
X, y = load_bulldozer(n)

gen_topk_figs(X,y,kfolds=5,n_trials=5,dataset="bulldozer",title="Bulldozer auction prices",
              catcolnames={'AC', 'ModelID', 'auctioneerID'},
              yrange=(5000,20000), yunits="$")
