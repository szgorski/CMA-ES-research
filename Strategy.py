from typing import Callable, Any
from numbers import Real
import numpy as np


class Strategy(object):
    def __init__(self,
                 function: Callable,
                 x_initial: Any,
                 max_limit: int,
                 limit_evaluations: bool,
                 seed: int | None = None):

        # problem initialization
        self.func = function
        self.x_init = x_initial
        self.dim = 0
        self.rand = np.random.default_rng(seed)

        if isinstance(x_initial, Real):
            self.dim = 1
        elif isinstance(x_initial, np.ndarray):
            self.dim = len(x_initial)
        else:
            raise TypeError

        self.max_iter = None
        self.max_eval = None
        if limit_evaluations:
            self.max_eval = max_limit
        else:
            self.max_iter = max_limit

        # results
        self.best_x = None
        self.best_value = None
