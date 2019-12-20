from multiprocessing import  Pool
from functools import partial
import numpy as np
import pandas as pd

def doit(df):
    print(id(df)) # gets diff id so must be in diff process and sharing?
    print()

def parallelize(df, func, num_of_processes=8):
    pool = Pool(num_of_processes)
    data = pool.map(func, [df,df])
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def go():
    df = pd.DataFrame([[1, 2],[3, 4]])
    parallelize(df, doit, 8)

go()