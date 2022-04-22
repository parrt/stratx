import support
from stratx import importances, plot_importances
import matplotlib.pyplot as plt
import numpy as np
from support import synthetic_files

for dataset in synthetic_files:
    print(f"Plot stability for {dataset}")
    np.random.seed(1)
    X, y, X_train, X_test, y_train, y_test = support.load_dataset(dataset, targetname='response')

    print(X.shape)

    I = importances(X, y, bootstrap=False, n_trials=30, subsample_size=.75, n_jobs=4)

    print(I)

    # Don't need for continuous x_i since importance == impact when
    # there are n unique values for n values
    # plot_importances(I[0:10], imp_range=(0, 0.4), sortby='Importance')
    #plt.savefig(f"../images/{dataset}-stability-importance.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    # plt.close()

    plot_importances(I[0:10], imp_range=(0, 0.4), sortby='Impact')
    plt.savefig(f"../images/{dataset}-stability-impact.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()