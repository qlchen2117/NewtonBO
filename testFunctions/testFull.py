import numpy as np
import torch
from botorch.test_functions import Ackley, Levy, Griewank
from pathlib import Path
# import logging
# logging.getLogger().setLevel(logging.INFO)


def testFull(func_name, N_EPOCH, DIM, EM_DIM, N_ITERACTIONS, MAX_TIME, N_INIT, BATCH_SIZE, methods):
    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE
    if func_name == 'Ackley':
        func = Ackley(dim=DIM)
        func.bounds[0, :].fill_(-5)
        func.bounds[1, :].fill_(10)
    elif func_name == 'Levy':
        func = Levy(dim=DIM)
        func.bounds[0, :].fill_(-5)
        func.bounds[1, :].fill_(10)
    elif func_name == 'Griewank':
        func = Griewank(DIM)
        func.bounds[0, :].fill_(-300.)
        func.bounds[1, :].fill_(600.)
    else:
        raise NotImplementedError

    res_p = Path(f"results/{type(func).__name__}-full")
    if not res_p.exists():
        res_p.mkdir(parents=True)

    lb, ub = func.bounds
    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return func(lb + (ub - lb) * x).item() * -1

    def eval_objective4min(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return func(lb + (ub - lb) * x).item()

    def eval_objective_wrap(parameterization):  # wrapper for alebo
        """x is assumed to be in [0, 1]^d"""
        x = torch.tensor([parameterization.get(f"x{i}") for i in range(DIM)]).to(lb)
        x = lb + (ub - lb) * x
        y = func(x).item() * -1 # Flip the value for minimization
        return {"objective": (y, 0.0)}

    def eval_objective4min_np(x):  # wrapper for hebo
        """x is assumed to be in [0, 1]^d"""
        x = torch.tensor(x).to(lb)
        x = lb + (ub - lb) * x
        y = func(x).item()
        return y

    def eval_objective4max_np(x: np.ndarray):
        """x is assumed to be in [0, 1]^d"""
        x = torch.tensor(x).to(lb)
        x = lb + (ub - lb) * x
        return func(x).item() * -1 # Flip the value

    if 'saasbo' in methods:
        from baseline.saasbo import saasbo
        for i in range(N_EPOCH):
            _, Y = saasbo(eval_func=eval_objective, ndims=DIM, n_iterations=N_ITERACTIONS, n_init=N_INIT, batch_size=BATCH_SIZE)
            Y_np = -1 * Y.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-saasbo{i}.npy", Y_np.ravel())

    if 'rembo' in methods:
        from baseline.rembo import rembo
        for i in range(N_EPOCH):
            _, Y, clock = rembo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE)
            Y_np = -1 * Y.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-rembo{i}.npy", Y_np.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-rembo-clock{i}.npy", clock)

    if 'alebo' in methods:
        from baseline.alebo_sim.alebo_wrap import alebo
        for i in range(N_EPOCH):
            _, Y, clock = alebo(eval_objective4min, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE)
            Y_np = Y.cpu().numpy()
            # wallclocks[i] = clock.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-alebo{i}.npy", Y_np.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-alebo-clock{i}.npy", clock)

    if 'hebo' in methods:
        from BO.hebo_wrapper import hebo_optimize
        for i in range(N_EPOCH):
            print(f"HEBO epoch{i}...")
            _, Y_np = hebo_optimize(eval_objective4min_np, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-hebo{i}.npy", Y_np.ravel())

    if 'bo_moo' in methods:
        from BO.bo import bo
        for i in range(N_EPOCH):
            _, Y = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE, use_input_warp=False, opt_acq='MACE')
            Y_np = Y.cpu().numpy() * -1
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-bo_moo{i}.npy", Y_np.ravel())

    if 'gp-ei' in methods:
        from BO.bo import bo
        for i in range(N_EPOCH):
            _, Y, clock = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE, acq='EI')
            Y_np = Y.cpu().numpy() * -1
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-gp-ei{i}.npy", Y_np.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-gp-ei-clock{i}.npy", clock)

    if 'gp-pes' in methods:
        from BO.bo import bo
        for i in range(N_EPOCH):
            _, Y, _ = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE, acq='PES', optimizer='CMA-ES')
            Y_np = Y.cpu().numpy() * -1
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-gp-pes{i}.npy", Y_np.ravel())

    # from HDBO.kdr_bo.kdr_bo import kdr_bo
    # for i in range(N_EPOCH):
    #     _, Y, clock = kdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    #     wallclocks[i] = clock.cpu().numpy()
    # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-kdr_bo.npy"), store_data)
    # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-kdr_bo-clock.npy"), wallclocks)

    # from HDBO.mkdr_bo.mkdr_bo import mkdr_bo
    # for i in range(N_EPOCH):
    #     _, Y, clock = mkdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    #     wallclocks[i] = clock.cpu().numpy()
    # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-mkdr_bo.npy"), store_data)
    # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-mkdr_bo-clock.npy"), wallclocks)

    # from HDBO.mkdr_bo.mkdr_bo_mtgp import mkdr_bo_mtgp
    # for i in range(N_EPOCH):
    #     _, Y = mkdr_bo_mtgp(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-mtgp.npy"), store_data)

    if 'turbo1' in methods:
        from baseline.turbo import Turbo1
        for i in range(N_EPOCH):
            turbo1 = Turbo1(
                f=eval_objective4min_np, lb=np.zeros(DIM), ub=np.ones(DIM),
                n_init=N_INIT, max_evals=TOTAL_TRIALS, batch_size=BATCH_SIZE, verbose=False)
            turbo1.optimize()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-turbo1-{i}.npy", turbo1.fX.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-turbo1-clock{i}.npy", np.array(turbo1.wallclocks))

    if 'turbom' in methods:
        from baseline.turbo import TurboM
        for i in range(N_EPOCH):
            turbo_m = TurboM(
                f=eval_objective4min_np, lb=np.zeros(DIM), ub=np.ones(DIM), n_init=N_INIT // BATCH_SIZE,
                max_evals=TOTAL_TRIALS, n_trust_regions=BATCH_SIZE, batch_size=BATCH_SIZE, verbose=False)
            clock = turbo_m.optimize(MAX_TIME)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-turbom{i}.npy", turbo_m.fX.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-turbom-clock{i}.npy", clock)

    if 'newton-bo' in methods:
        from HDBO.turboD import turboD
        for i in range(N_EPOCH):
            _, Y, clock = turboD(eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, max_time=MAX_TIME, batch_size=BATCH_SIZE)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-newton-bo{i}.npy", Y.view(-1).cpu().numpy())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-newton-bo-clock{i}.npy", clock)

    if 'newton-bo-rand' in methods:
        from HDBO.turboD import turboD
        for i in range(N_EPOCH):
            _, Y, clock = turboD(eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, max_time=MAX_TIME, batch_size=BATCH_SIZE, restart_strategy='random')
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-newton-bo-rand{i}.npy", Y.view(-1).cpu().numpy())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-newton-bo-rand-clock{i}.npy", clock)

    if 'turbo-d-drop' in methods:
        from HDBO.turboD import turboD_drop
        for i in range(N_EPOCH):
            _, Y = turboD_drop(
                eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, batch_size=BATCH_SIZE, active_dim=50
            )
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-turboD-drop{i}.npy", Y.view(-1).cpu().numpy())

    if 'gibo' in methods:
        from baseline.gibo.gibo import gibo
        for i in range(N_EPOCH):
            _, Y, clock = gibo(objective=eval_objective, dim=DIM, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, n_init=N_INIT, batch_size=BATCH_SIZE, optimizer4mll='scipy')
            Y_np = -1 * Y.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-gibo{i}.npy", Y_np.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-gibo-clock{i}.npy", clock)

    if 'bo-mpd' in methods:
        from baseline.local_bo_mpd.local_bo_mpd import local_bo_mpd
        for i in range(N_EPOCH):
            _, Y = local_bo_mpd(objective=eval_objective, dim=DIM, n_iterations=N_ITERACTIONS, n_init=N_INIT, batch_size=BATCH_SIZE)
            Y_np = -1 * Y.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-mpd{i}.npy", Y_np.ravel())

    if 'sobol' in methods:
        from BO.sobol import sobol
        for i in range(N_EPOCH):
            _, Y, clock = sobol(eval_objective, ndims=DIM, total_trials=TOTAL_TRIALS)
            Y_np = -1 * Y.cpu().numpy()
            # wallclocks[i] = clock.cpu().numpy()
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-sobol{i}.npy", Y_np.ravel())
            # np.save(res_p / f"{type(func).__name__}-full-D{DIM}-sobol-clock.npy"), wallclocks)

    if 'sir-bo' in methods:
        from baseline.sir_bo.sir_bo import sir_bo
        for i in range(N_EPOCH):
            _, Y, clock = sir_bo(obj_fct=eval_objective4min_np, dim=EM_DIM, high_dim=DIM, init_size=N_INIT, n_iter=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-sir_bo{i}.npy", Y.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-sir_bo-clock{i}.npy", clock)

    if 'ksir-bo' in methods:
        from baseline.sir_bo.ksir_bo import ksir_bo
        for i in range(N_EPOCH):
            _, Y, clock = ksir_bo(obj_fct=eval_objective4min_np, dim=EM_DIM, high_dim=DIM, init_size=N_INIT, n_iter=N_ITERACTIONS, max_time=MAX_TIME, batch_size=BATCH_SIZE)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-ksir_bo{i}.npy", Y.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-ksir_bo-clock{i}.npy", clock)

    if 'add-gp-ucb' in methods:
        from baseline.add_bo_quasi_nt.BOLibkky.addGPBO import add_gp_bo
        from baseline.add_bo_quasi_nt.BOLibkky.preprocessDecomposition import HyperParam
        for i in range(N_EPOCH):
            _, boVals, clock = add_gp_bo(
                eval_objective,
                bounds=np.array([[0., 1.]]*DIM),
                num_iterations=N_ITERACTIONS,
                max_time=MAX_TIME,
                n_init=N_INIT,
                batch_size=BATCH_SIZE,
                params=HyperParam(DIM, 4, True)
            )
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-add_bo{i}.npy", boVals * -1)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-add_bo-clock{i}.npy", clock)

    if 'cmaes' in methods:
        from baseline.cmaes import cmaes
        for i in range(N_EPOCH):
            _, Y, clock = cmaes(objective=eval_objective4min_np, dim=DIM, n_iterations=N_ITERACTIONS, max_time=MAX_TIME, n_init=N_INIT, batch_size=BATCH_SIZE)
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-cmaes{i}.npy", Y.ravel())
            np.save(res_p / f"{type(func).__name__}-full-D{DIM}-cmaes-clock{i}.npy", clock)

# # from HDBO.mybo import manifold_bo
# for i in range(N_EPOCH):
#     _, Y = manifold_bo(eval_objective, dim=DIM, emb_dim=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
#     Y_np = -1 * Y.cpu().numpy()
#     store_data[i] = Y_np.ravel()
# np.save(res_p / f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-manifold_bo.npy"), store_data)
