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
    return time_func


def torch_permute(contiguous):
    values = torch.randn(5000, 5000)
    @torch.compile
    def time_func():
        res = torch.permute(values, (1, 0))
        if contiguous:
            res = res.contiguous()
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


def time_results(prefix, func):
    print(f"{prefix}: {timeit.timeit(func, number=N_ITER) / N_ITER * 100000:.1f} micro seconds")


if __name__ == "__main__":
    N_ITER = 50

    jax_function_fast()()
    torch_function_fast()()
    
    time_results("Numpy", numpy_function())
    time_results("Numpy permute non-contiguous", numpy_permute(False))
    time_results("Numpy permute contiguous", numpy_permute(True))
    time_results("Torch slow", torch_function_slow())
    time_results("Torch fast", torch_function_fast())
    time_results("Torch permute non-contiguous", torch_permute(False))
    time_results("Torch permute contiguous", torch_permute(True))
    time_results("Jax slow", jax_function_slow())
    time_results("Jax fast", jax_function_fast())
