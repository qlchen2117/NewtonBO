## Overview

This is the code-release for the Newton-BO algorithm from ***Enhancing Trust-Region Bayesian Optimization via Newton Methods*** appearing in ECAI 2025.

Note that Newton-BO is a **minimization** algorithm, so please make sure you reformulate potential maximization problems.

## Benchmark functions

### Synthetic functions
The original code for synthetic functions is available at BoTorch.

Dependencies: ```conda-forge::pytorch==2.3```, ```conda-forge::gpytorch==1.11```, ```conda-forge::botorch==0.10.0```

### Weighted Lasso Tuning
The original code for the weighted Lasso tuning problem is available at https://github.com/ksehic/LassoBench.
The goal of the problem is to tune the Lasso (Least Absolute Shrinkage and Selection Operator) regression models.

### Vehicle Design
The original code for the vehicle design problem (Mopta08) is available at https://github.com/LeoIV/BAxUS/blob/main/baxus/benchmarks/real_world_benchmarks.py. 
The goal of the problem is to minimize the mass of a vehicle characterized by 124 design variables describing materials, gauges, and vehicle shape.

### Robot pushing
The original code for the robot pushing problem is available at https://github.com/zi-w/Ensemble-Bayesian-Optimization. We have made the following changes to the code when running our experiments:

1. We turned off the visualization, which speeds up the function evaluations.

Dependencies: ```conda-forge::pygame``` (Optional), ```conda-forge::box2d-py```

### Rover
The original code for the rover problem is available at https://github.com/zi-w/Ensemble-Bayesian-Optimization. We used the large version of the problem, which has 60 dimensions.

Dependencies: ```numpy```, ```scipy```

### MuJoCo Locomotion
The goal of the problem is to learn a linear policy that maximizes the accumulative reward in MuJoCo tasks, similar to reinforcement learning, akin to reinforcement learning.


Dependencies: ```conda-forge::gymnasium==1.0.0```, ```conda-forge::mujoco==3.2.0```, ```conda-forge::imageio==2.36.1```

## Examples
```
python main_fun.py  # The example on running synthetic functions

python main_lasso.py  # The example on running weighted Lasso tuning benchmark

python main_mopta.py  # The example on running the vehicle design benchmark

python main_push.py  # The example on running the robot pushing benchmark

python main_Rover.py  # The example on running the rover benchmark

python main_RL.py  # The example on running the MuJoCo benchmark
```