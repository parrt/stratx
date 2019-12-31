from timeit import default_timer as timer

from stratx.featimp import *
from stratx.partdep import *

R = pd.read_feather("/tmp/boston.feather")
R = R.set_index('index')

fig, ax = plt.subplots(1,1)

plot_topk(R, ax)
plt.tight_layout()
#plt.savefig("/Users/parrt/Desktop/t.png", bbox_inches="tight", pad_inches=0)
plt.show()
