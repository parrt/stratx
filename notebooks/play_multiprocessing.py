from multiprocessing import  Pool
from functools import partial
import numpy as np
import pandas as pd
from joblib import parallel_backend, Parallel, delayed

def doit(df):
    print(id(df)) # gets diff id so must be in diff process and sharing?
    return np.sum(df)

def parallelize(df, func, num_of_processes=8):
    pool = Pool(num_of_processes)
    data = pool.map(func, [df,df])
    pool.close()
    pool.join()
    return data


def go():
    df = pd.DataFrame([[1, 2],[3, 4]])
    print(df)
    #parallelize(df, doit, 8)

    with parallel_backend('threading', n_jobs=2):
        print(Parallel()(delayed(doit)(df.iloc[i]) for i in range(2)))

go()