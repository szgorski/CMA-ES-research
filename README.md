# Evolution Strategies for Continuous Optimization

## Overview

This project implements and evaluates several variants of Evolution Strategies (ES) for continuous optimization problems, with a focus on benchmarking performance on standardized test suites.

The implemented algorithms include:

- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Matrix Adaptation Evolution Strategy (MA-ES)
- Increasing Population Size MA-ES (IPOP-MA-ES)

The framework supports experimentation on both **CEC 2021 benchmark functions** and the **BBOB (Black-Box Optimization Benchmarking) suite**, enabling comparative analysis across multiple dimensions and stochastic runs. 

Algorithms operate shared `Strategy` base class. The program supports for both evaluation-limited and iteration-limited optimization, as well as adaptive step-size control.

## Theoretical Background

Evolution Strategies (ES) are stochastic, population-based optimization algorithms designed for black-box optimization. They iteratively improve candidate solutions by sampling from parameterized probability distributions.

### CMA-ES

CMA-ES adapts a full covariance matrix to capture dependencies between variables, enabling efficient navigation of ill-conditioned and non-separable problems.

### MA-ES

MA-ES replaces covariance matrix adaptation with a transformation matrix, reducing computational complexity while maintaining adaptation capabilities.

### IPOP-MA-ES

IPOP-MA-ES introduces a restart strategy fora given population size, promising improved global search capabilities and robustness against local optima.

## Experimental Setup

Experiments are conducted using:

- Multiple benchmark functions (CEC 2021, BBOB)
- Multiple dimensionalities (10 and 20)
- Common, pre-set evaluation budget (50,000 evaluations)
- Multiple, controlled random seeds for statistical robustness

Each algorithm is evaluated across multiple runs, and results are aggregated for analysis. Configuration can be set in `CEC_2021_test.py` and `bbob_test.py`, respectively.

## Evaluation Methodology

Performance is assessed using two complementary metrics:

1. **Normalized Error (NE)**  
   Measures deviation from the known global optimum.

2. **Ranking-Based Score**  
   Compares algorithms based on relative performance across functions.

The final score is computed as a combination of both metrics, providing a balanced evaluation of accuracy and consistency.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running CEC Benchmark

Inside `CEC_2021_test.py`, setup desired configuration and import selected strategy:

```python
from IPOP_MA_ES import IPOP_MAES

def f_min(function: Callable,
          x_initial: Any,
          seed: int):
    es = IPOP_MAES(function, x_initial, budget, count_evaluations, seed)
    ...
```

then run:

```bash
python CEC_2021_test.py
```

### Running BBOB Benchmark

Inside `bbob_test.py`, setup desired configuration and import selected strategy:

```python
from IPOP_MA_ES import IPOP_MAES

def f_min(function: Callable,
          x_initial: Any,
          seed: int):
    es = IPOP_MAES(function, x_initial, budget, count_evaluations, seed)
    ...
```

then run:

```bash
python bbob_test.py
```
