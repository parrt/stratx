from support import *
import numpy as np

np.random.seed(1)

gen_topk_figs(n_trials=1,dataset="bulldozer",targetname='SalePrice',
              title="Bulldozer auction prices",
              catcolnames={'AC', 'ModelID', 'auctioneerID'},
              min_samples_leaf=20,
              cat_min_samples_leaf=10,
              yrange=(1000,20000), yunits="$")
