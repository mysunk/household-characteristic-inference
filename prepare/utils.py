""" Define util functions
"""

import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.2f} seconds')
        return result
    return timeit_wrapper

def class_decorator(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and not attr.startswith('__'):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
