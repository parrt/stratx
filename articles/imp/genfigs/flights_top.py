from support import *

use_oob=False
metric = mean_absolute_error
n = 20_000 # 30k crashes shap so try 20k

X, y, _ = load_flights(n=n)

R, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         catcolnames={'AIRLINE',
                                      'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                                      'FLIGHT_NUMBER',
                                      'DAY_OF_WEEK', 'dayofyear'},
                         metric=mean_squared_error,
                         min_slopes_per_x=10, # a bit less than usual (gridsearch showed how to get value)
                         use_oob=use_oob,
                         top_features_range=(1, 8))

plot_importances(our_I.iloc[0:8], imp_range=(0, .4), width=4.5,
                 title="Flight arrival delay StratImpact importances")
plt.tight_layout()
plt.savefig("../images/flights-features.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

plot_importances(rf_I.iloc[0:8], imp_range=(0, .4), width=4.5,
                 title="Flight arrival delay RF SHAP importances")
plt.tight_layout()
plt.savefig("../images/flights-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

print(R)

R = R.reset_index(drop=True)
R.reset_index().to_feather("/tmp/flights.feather")

fig, ax = plt.subplots(1,1,figsize=(4,3.5))
plot_topk(R, ax, k=8)
ax.set_ylabel("Training MAE (minutes)")
ax.set_title("Flight arrival delay")
plt.tight_layout()
plt.savefig("../images/flights-topk.pdf", bbox_inches="tight", pad_inches=0)
plt.show()