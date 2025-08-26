import cma
import numpy as np
import time
from utils import latin_hypercube


def cmaes(objective, dim, n_iterations, max_time, n_init, batch_size):
    startT = time.monotonic()
    xs = latin_hypercube(n_init, dim)
    ys = np.array([objective(xi) for xi in xs])
    X_cma, Y_cma = [xs], [ys]  # Append data
    wallclocks = [time.monotonic() - startT] * n_init

    es = cma.CMAEvolutionStrategy(  # create the CMA-ES optimizer
        x0=np.ones(dim) * .5,
        sigma0=1.,
        inopts={"bounds": [0, 1], "popsize": batch_size, 'maxiter': n_iterations+1},  # iter1 is random
    )
    es.ask()
    es.tell(xs, ys)
    while not es.stop() and time.monotonic() - startT < max_time:
        xs = es.ask()
        ys = np.array([objective(xi) for xi in xs])  # shape(popsize,)
        es.tell(xs, ys)
        X_cma.append(xs)  # Append data
        Y_cma.append(ys)
        wallclocks.extend([time.monotonic() - startT]*batch_size)
    return np.vstack(X_cma), np.hstack(Y_cma), np.array(wallclocks)
