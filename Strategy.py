from Function import *


class Strategy(object):
    def __init__(self, function: Function, max_iterations: int):
        self.func = function
        self.max_iter = max_iterations
