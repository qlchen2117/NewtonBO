
""" Run Rover benchmarks. Maximization problems. """
# import warnings
# warnings.filterwarnings('ignore')
# import logging
# logging.disable(logging.INFO)
import numpy as np
from torch import Tensor
from benchmarks.rover_function import create_problem

# # Save Results
from pathlib import Path
RES_PATH = Path("results/rover/")
if not RES_PATH.exists():
    RES_PATH.mkdir(parents=True)


def testRover(N_EPOCH: int, N_ITERACTIONS: int, N_INIT: int, BATCH_SIZE: int, methods):
    func_name = 'Rover'
    problem = create_problem()
    x_range = problem.get_range()
    lb, ub = x_range[0], x_range[1]
    DIM = len(lb)
    EM_DIM = 10
    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE

    def eval_objective(x: Tensor):
        x_np = lb + (ub - lb) * x.numpy()
        return problem(x_np)

    def eval_objective4min(x: Tensor):
        x_np = lb + (ub - lb) * x.numpy()
        return problem(x_np) * -1  # Flip the value for minimization

    def eval_objective4alebo(parameterization):  # wrapper for alebo
        """x is assumed to be in [0, 1]^d"""
        x_np = np.array([parameterization.get(f"x{i}") for i in range(DIM)])
        y = problem(lb + (ub - lb) * x_np)
        return {"objective": (y, 0.0)}

    def eval_objective4min_np(x: np.ndarray):  # wrapper for hebo
        """x is assumed to be in [0, 1]^d"""
        return problem(lb + (ub - lb) * x) * -1 # Flip the value for minimization


    store_data = np.empty((N_EPOCH, TOTAL_TRIALS))

    if 'saasbo' in methods:
        from HDBO.saasbo import saasbo
        for i in range(N_EPOCH):
            _, Y = saasbo(eval_func=eval_objective, ndims=DIM, n_iterations=N_ITERACTIONS, n_init=N_INIT, batch_size=BATCH_SIZE)
            Y_np = -1 * Y.cpu().numpy()
            store_data[i] = Y_np.ravel()
        np.save(RES_PATH / func_name+f"-D{DIM}-saasbo.npy", store_data)

    if 'rembo' in methods:
        from HDBO.rembo import rembo
        for i in range(N_EPOCH):
            _, Y = rembo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
            Y_np = -1 * Y.cpu().numpy()
            store_data[i] = Y_np.ravel()
            # wallclocks[i] = clock.cpu().numpy()
        np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-rembo.npy", store_data)

    if 'alebo' in methods:
        from HDBO.alebo_sim.alebo_wrap import alebo
        for i in range(N_EPOCH):
            _, Y = alebo(eval_objective4min, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
            Y_np = Y.cpu().numpy()
            # store_data[i] = Y_np.ravel()
            # wallclocks[i] = clock.cpu().numpy()
            np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-alebo{i}.npy", Y_np.ravel())

    if 'turbo1' in methods:
        from HDBO.turbo import Turbo1
        for i in range(N_EPOCH):
            turbo1 = Turbo1(
                f=eval_objective4min_np, lb=np.zeros(DIM), ub=np.ones(DIM),
                n_init=N_INIT, max_evals=TOTAL_TRIALS, batch_size=BATCH_SIZE, verbose=False)
            turbo1.optimize()
            store_data[i] = turbo1.fX.ravel()[:TOTAL_TRIALS]  # Observed values
            # wallclocks[i] = np.array(turbo1.wallclocks[:TOTAL_TRIALS])
        np.save(RES_PATH / func_name+f"-D{DIM}-turbo1.npy", store_data)

    if 'turbom' in methods:
        from HDBO.turbo import TurboM
        for i in range(N_EPOCH):
            turbo_m = TurboM(
                f=eval_objective4min_np, lb=np.zeros(DIM), ub=np.ones(DIM), n_init=N_INIT // BATCH_SIZE,
                max_evals=TOTAL_TRIALS, n_trust_regions=BATCH_SIZE, batch_size=BATCH_SIZE, verbose=False)
            turbo_m.optimize()
            store_data[i] = turbo_m.fX.ravel()[:TOTAL_TRIALS]  # Observed values
        np.save(RES_PATH / func_name+f"-D{DIM}-turbom.npy", store_data)

    if 'turbo-d' in methods:
        from HDBO.turboD import turboD
        for i in range(N_EPOCH):
            _, Y = turboD(eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, batch_size=BATCH_SIZE)
            store_data[i] = Y.view(-1).cpu().numpy()[:TOTAL_TRIALS]
        np.save(RES_PATH / func_name+f"-D{DIM}-turboD.npy", store_data)

    if 'turbo-d-drop' in methods:
        from HDBO.turboD import turboD_drop
        for i in range(N_EPOCH):
            _, Y = turboD_drop(
                eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, n_trust_regions=BATCH_SIZE,
                batch_size=1, active_dim=50
            )
            store_data[i] = Y.view(-1).cpu().numpy()[:TOTAL_TRIALS]
        np.save(RES_PATH / func_name+f"-D{DIM}-turboD-drop.npy", store_data)

    if 'gibo' in methods:
        from HDBO.gibo.gibo import gibo
        for i in range(N_EPOCH):
            _, Y = gibo(objective=eval_objective, dim=DIM, n_iterations=N_ITERACTIONS, n_init=N_INIT, batch_size=BATCH_SIZE, optimizer4mll='scipy')
            Y_np = -1 * Y.cpu().numpy()
            # store_data[i] = Y_np.ravel()[:TOTAL_TRIALS]
            np.save(RES_PATH / func_name+f"-D{DIM}-gibo{i}.npy", Y_np.ravel()[:TOTAL_TRIALS])

    # from HDBO.kdr_bo.kdr_bo import kdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = kdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-kdr_bo.npy", store_data)

    # from HDBO.mkdr_bo.mkdr_bo import mkdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = mkdr_bo(
    #         eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS,
    #         gp_lr=0.01, batchgp_lr=0.1, acq_restart=5, optim_iter=3
    #     )
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-mkdr_bo.npy", store_data)

    if 'sobol' in methods:
        from BO.sobol import sobol
        for i in range(N_EPOCH):
            _, Y, clock = sobol(eval_objective, ndims=DIM, total_trials=TOTAL_TRIALS)
            Y_np = -1 * Y.cpu().numpy()
            store_data[i] = Y_np.ravel()
            # wallclocks[i] = clock.cpu().numpy()
        np.save(RES_PATH / func_name+f"-D{DIM}-sobol.npy", store_data)

    if 'sir-bo' in methods:
        from HDBO.sir_bo.sir_bo import sir_bo
        for i in range(N_EPOCH):
            _, Y = sir_bo(eval_objective4min_np, EM_DIM, DIM, N_INIT, N_ITERACTIONS, BATCH_SIZE)
            store_data[i] = Y.ravel()
        np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-sir_bo.npy", store_data)

    if 'ksir-bo' in methods:
        from HDBO.sir_bo.ksir_bo import ksir_bo
        for i in range(N_EPOCH):
            _, Y = ksir_bo(eval_objective4min_np, EM_DIM, DIM, N_INIT, N_ITERACTIONS, BATCH_SIZE)
            store_data[i] = Y.ravel()
        np.save(RES_PATH / func_name+f"-D{DIM}-d{EM_DIM}-ksir_bo.npy", store_data)

    if 'add-gp-ucb' in methods:
        from HDBO.add_bo_quasi_nt.BOLibkky.addGPBO import add_gp_bo
        from HDBO.add_bo_quasi_nt.BOLibkky.preprocessDecomposition import HyperParam
        for i in range(N_EPOCH):
            _, boVals = add_gp_bo(
                eval_objective,
                bounds=np.array([[0., 1.]]*DIM),
                num_iterations=N_ITERACTIONS,
                n_init=N_INIT,
                batch_size=BATCH_SIZE,
                params=HyperParam(DIM, 4, True)
            )
            store_data[i] = boVals * -1
            # wallclocks[i] = clock
        np.save(RES_PATH / func_name+f"-D{DIM}-add_bo.npy", store_data)


    # from BO.bo import bo
    # _, Y = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS,
    #           use_input_warp=True, use_moo=True)
    # Y_np = Y.cpu().numpy() * -1
    # Y_bo_moo[i] = Y_np.squeeze()

    if 'gp-ei' in methods:
        from BO.bo import bo
        for i in range(N_EPOCH):
            _, Y = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE, acq='EI')
            Y_np = Y.cpu().numpy() * -1
            # store_data[i] = Y_np.ravel()
            np.save(RES_PATH / func_name+f"-D{DIM}-gp-ei{i}.npy", Y_np.ravel())

    if 'cmaes' in methods:
        from baseline.cmaes import cmaes
        for i in range(N_EPOCH):
            _, Y, _ = cmaes(objective=eval_objective4min_np, dim=DIM, n_iterations=N_ITERACTIONS, max_time=float("inf"), n_init=N_INIT, batch_size=BATCH_SIZE)
            np.save(RES_PATH / func_name+f"-D{DIM}-cmaes{i}.npy", Y.ravel())

    # from BO.hebo_wrapper import hebo_optimize
    # _, Y = hebo_optimize(eval_objective4hebo, ndims=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    # Y_hebo[i] = Y.squeeze()
