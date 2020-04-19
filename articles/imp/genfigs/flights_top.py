from support import *

np.random.seed(1) # good results

n = 25_000
X, y, df_flights = load_flights(n=n)
# df_flights = pd.read_csv("flights20k.csv")

gen_topk_figs(X,y,kfolds=5,n_trials=5,dataset="flights",title="Flight arrival delay",
              catcolnames={'AIRLINE',
                           'ORIGIN_AIRPORT',
                           'DESTINATION_AIRPORT',
                           'FLIGHT_NUMBER',
                           'DAY_OF_WEEK'},
              cat_min_samples_leaf=2,  # reduce as there are lots of other cat vars
              yrange=(5,30), yunits="min")

