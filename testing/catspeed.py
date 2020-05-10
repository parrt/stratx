from stratx.partdep import *
from articles.pd.support import *
from timeit import default_timer as timer

import numpy as np

from test_catmerge import stratify_cats

def speed_ModelID():
    "I believe none of this is in the JIT path; repeated runs are same speed"
    np.random.seed(1)

    n = 20_000
    min_samples_leaf = 5
    X,y = load_bulldozer(n=n)

    leaf_deltas, leaf_counts, ignored = \
        stratify_cats(X,y,colname="ModelID",min_samples_leaf=min_samples_leaf)

    start = timer()
    _, _, merge_ignored = \
        avg_values_at_cat(leaf_deltas, leaf_counts, max_iter=10)
    stop = timer()

    nunique = len(np.unique(X['ModelID']))
    print(f"n={n}, unique cats {nunique}, min_samples_leaf={min_samples_leaf}, merge_ignored={merge_ignored}: avg_values_at_cat {stop - start:.3f}s")


if __name__ == '__main__':
    speed_ModelID()
