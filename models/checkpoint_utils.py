import torch
import functools

def method_dummy_wrapper(method):
    @functools.wraps(method)
    def wrapped(dummy, *args, **kwargs):
        return method(*args, **kwargs)
    return wrapped