import copy
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
        self.func = None   # function
        self.x_init = x_initial
        self.dim = 0
        self.rand = np.random.default_rng(seed)

        if isinstance(x_initial, Real):
            self.dim = 1
        elif isinstance(x_initial, np.ndarray):
            self.dim = len(x_initial)
        else:
            raise TypeError

        self.limit_eval = limit_evaluations
        self.max_iter = None
        self.max_eval = None
        if limit_evaluations:
            self.max_eval = max_limit
        else:
            self.max_iter = max_limit

        # setup evaluation function for bbob or CEC
        self.lb = None
        self.ub = None
        if hasattr(function, 'evaluate'):       # CEC function
            self.func = function.evaluate
            self.lb = function.lb
            self.ub = function.ub
        else:                                   # bbob function
            self.func = function
            self.lb = function.lower_bounds
            self.ub = function.upper_bounds

        # results
        self.best_x = None
        self.best_value = None
