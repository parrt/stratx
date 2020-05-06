from support import *

np.random.seed(1)

gen_topk_figs(n_trials=1,dataset="flights",targetname='ARRIVAL_DELAY',
              title="Flight arrival delay",
              catcolnames={'AIRLINE',
                           'ORIGIN_AIRPORT',
                           'DESTINATION_AIRPORT',
                           'FLIGHT_NUMBER',
                           'TAIL_NUMBER',
                           'DAY_OF_WEEK'},
              min_samples_leaf=20,
              cat_min_samples_leaf=10,
              yrange=(0,30), yunits="min")

