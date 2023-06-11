class Function(object):
    def __init__(self, dimension: int, lower_bound: int | float, upper_bound: int | float):
        self.dim = dimension
        self.l_bound = lower_bound
        self.u_bound = upper_bound

    def evaluate(self, x):
        raise NotImplementedError
