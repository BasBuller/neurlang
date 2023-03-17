import timeit

import numpy as np
import jax
import torch


def numpy_function():
    values = np.random.randn(5000, 5000)
    def time_func():
        res = values * values + values * 2
    return time_func


def numpy_permute(contiguous):
    values = np.random.randn(5000, 5000)
    def time_func():
        res = np.transpose(values, (1, 0))
        if contiguous:
            res = np.ascontiguousarray(res)
    return time_func


def torch_function_slow():
    values = torch.randn(5000, 5000)
    def time_func():
        res = values * values + values * 2
    return time_func


def torch_function_fast():
    values = torch.randn(5000, 5000)
    @torch.compile
    def time_func():
        res = values * values + values * 2
    time_func()
    return time_func


def torch_permute(contiguous):
    values = torch.randn(5000, 5000)
    @torch.compile
    def time_func():
        res = torch.permute(values, (1, 0))
        if contiguous:
            res = res.contiguous()
    time_func()
    return time_func


def jax_function_slow():
    values = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))
    def time_func():
        res = values * values + values * 2
    return time_func


def jax_function_fast():
    values = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))
    @jax.jit
    def time_func():
        res = values * values + values * 2
    _ = time_func()
    return time_func


if __name__ == "__main__":
    N_ITER = 10

    jax_function_fast()()
    torch_function_fast()()

    print("Numpy: ", timeit.timeit(numpy_function(), number=N_ITER) / N_ITER)
    print("Numpy permute non-contiguous: ", timeit.timeit(numpy_permute(contiguous=False), number=N_ITER) / N_ITER)
    print("Numpy permute contiguous: ", timeit.timeit(numpy_permute(contiguous=True), number=N_ITER) / N_ITER)
    print("Torch slow: ", timeit.timeit(torch_function_slow(), number=N_ITER) / N_ITER)
    print("Torch fast: ", timeit.timeit(torch_function_fast(), number=N_ITER) / N_ITER)
    print("Torch permute non-contiguous: ", timeit.timeit(torch_permute(contiguous=False), number=N_ITER) / N_ITER)
    print("Torch permute contiguous: ", timeit.timeit(torch_permute(contiguous=True), number=N_ITER) / N_ITER)
    print("Jax slow: ", timeit.timeit(jax_function_slow(), number=N_ITER) / N_ITER)
    print("Jax fast: ", timeit.timeit(jax_function_fast(), number=N_ITER) / N_ITER)
