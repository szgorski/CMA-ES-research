from CMA_ES import *
from MA_ES import *


class Sphere(Function):

    def __init__(self, dim, lb, ub):
        super().__init__(dim, lb, ub)

    def evaluate(self, x):
        x = self.l_bound + x * (self.u_bound - self.l_bound)
        re = sum(np.power(x[i], 2) for i in range(self.dim))
        return re


if __name__ == "__main__":
    TaskProb = Sphere(50, -50, 50)

    Task1 = MAES(TaskProb, 1000)
    Task1.calculate()

    Task2 = CMAES(TaskProb, 1000)
    Task2.calculate()
