import timeit

import numpy as np
import jax



def numpy_function():
    values = np.random.randn(5000, 5000)
    def time_func():
        res = values * values + values * 2
        return res
    return time_func


def jax_function_slow():
    values = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))
    def time_func():
        res = values * values + values * 2
        return res
    return time_func


def jax_function_fast():
    values = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))
    @jax.jit
    def time_func():
        res = values * values + values * 2
        return res
    _ = time_func()
    return time_func


if __name__ == "__main__":
    print("Numpy: ", timeit.timeit(numpy_function(), number=10) / 10)
    print("Jax slow: ", timeit.timeit(jax_function_slow(), number=10) / 10)
    print("Jax fast: ", timeit.timeit(jax_function_fast(), number=10) / 10)
