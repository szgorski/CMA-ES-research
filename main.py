from __future__ import division, print_function
from numpy.random import rand
from typing import Callable, Any
import cocoex
import cocopp
import os
import webbrowser

from CMA_ES import CMAES
# from MA_ES import MAES

# configuration
budget = 5
count_evaluations = False
output_folder = "scipy-optimize-fmin"
seed = 34763485763956  # either int or None


def f_min(function: Callable,
          x_initial: Any):
    es = CMAES(function, x_initial, budget, count_evaluations, seed)
    # es = MAES(function, x_initial, budget, count_evaluations, seed)
    es.calculate()

    return es.best_value


if __name__ == "__main__":
    # coco functions
    suite_name = "bbob"
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    # testing
    for problem in suite:
        problem.observe_with(observer)
        x_init = problem.initial_solution
        f_min(problem, x_init)
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                                     (problem.upper_bounds - problem.lower_bounds) / 2)
        minimal_print(problem, final=problem.index == len(suite) - 1)

    # post-process data
    cocopp.main(observer.result_folder)
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
