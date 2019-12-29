from timeit import default_timer as timer

from stratx.featimp import *
from stratx.partdep import *

I = pd.read_feather("/tmp/t.feather")
I = I.set_index('Feature')
#I.iloc[2,1] = .07
# print(I)
# plot_importances(I.iloc[0:4])
# plot_importances(I.iloc[0:8])
plot_importances(I.iloc[0:12])
# plot_importances(I)
# plot_importances(I, color='#FEE08F', dpi=150)
#plot_importances(I, imp_range=(0,0.5), width=2.5, vscale=1.35)
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/t.png")
plt.show()
