import numpy as np
import ray
from numba import jit


@jit(nopython=True)
def find_index(arr: np.ndarray, val):
    for i, v in enumerate(arr):
        if v == val:
            return i
    raise ValueError()


@ray.remote(scheduling_strategy="SPREAD", num_cpus=1)
def execute_function_on_list(func, arr, *args):
    return [(elem, func(elem, *args)) for elem in arr]
