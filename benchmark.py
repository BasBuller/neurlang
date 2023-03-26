import timeit

import numpy as np
import jax
import torch


SHAPE = (5000, 5000)


def numpy_function():
    values = np.random.randn(*SHAPE)
    def time_func():
        res = values * values + values * 2
        return res
    return time_func


def numpy_permute(contiguous):
    values = np.random.randn(*SHAPE)
    def time_func():
        res = np.transpose(values, (1, 0))
        if contiguous:
            res = np.ascontiguousarray(res)
        return res
    return time_func


def numpy_pad(contiguous):
    values = np.random.randn(*SHAPE)
    def time_func():
        res = np.pad(values, [(2, 2), (2, 2)])
        if contiguous:
            res = np.ascontiguousarray(res)
        return res
    return time_func


def torch_function(compile):
    values = torch.randn(*SHAPE)
    def time_func():
        res = values * values + values * 2
        return res
    if compile:
        time_func = torch.compile(time_func)
        time_func()
    return time_func


def torch_permute(contiguous, compile):
    values = torch.randn(*SHAPE)
    def time_func():
        res = torch.permute(values, (1, 0))
        if contiguous:
            res = res.contiguous()
        return res
    if compile:
        time_func = torch.compile(time_func)
        time_func()
    return time_func


def jax_function(compile):
    values = jax.random.normal(jax.random.PRNGKey(0), SHAPE)
    def time_func():
        res = values * values + values * 2
        return res
    if compile:
        time_func = jax.jit(time_func)
        time_func()
    return time_func


def time_results(prefix, func):
    print(f"{prefix}: {timeit.timeit(func, number=N_ITER) / N_ITER * 100000:.1f} micro seconds")


if __name__ == "__main__":
    N_ITER = 50

    time_results("Numpy", numpy_function())
    time_results("Numpy permute non-contiguous", numpy_permute(False))
    time_results("Numpy permute contiguous", numpy_permute(True))
    time_results("Numpy pad non-contiguous", numpy_pad(False))
    time_results("Numpy pad contiguous", numpy_pad(True))
    time_results("Torch slow", torch_function(False))
    time_results("Torch fast", torch_function(True))
    time_results("Torch permute non-contiguous - slow", torch_permute(False, False))
    time_results("Torch permute contiguous - slow", torch_permute(True, False))
    time_results("Torch permute non-contiguous - fast", torch_permute(False, True))
    time_results("Torch permute contiguous - fast", torch_permute(True, True))
    time_results("Jax slow", jax_function(False))
    time_results("Jax fast", jax_function(True))
