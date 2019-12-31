from support import *

R = pd.read_feather("/tmp/boston.feather")
R = R.set_index('index')

fig, ax = plt.subplots(1,1)
plot_topk(R, ax)
plt.tight_layout()
plt.show()


# R = pd.read_feather("/tmp/flights.feather")
# R = R.set_index('index')
#
# fig, ax = plt.subplots(1,1)
# plot_topk(R, ax, k=7)
# plt.tight_layout()
# plt.show()
