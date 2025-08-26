import numpy as np
# from botorch.test_functions import Ackley, Griewank, Levy
from pathlib import Path
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

# func, DIM = Hartmann(), 100, 6
# func, DIM, EM_DIM = , 30, 5
# func, DIM, EM_DIM = , 30, 9
# fig_p = Path(f"figures/{type(func).__name__}-full.png")

def plot_data(data, label):
    print(f"{label}: {data.shape}")
    data = np.minimum.accumulate(data, axis=1)
    f = data.mean(axis=0)
    sem = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
    x = np.arange(1, data.shape[-1]+1)
    ax.plot(x, f)
    ax.fill_between(x, f - 2. * sem, f + 2. * sem, alpha = 0.3, label=label)
    # ax.errorbar(x, f, yerr=sem, alpha=0.3, ls='')

def plotFunc_f(func_name, DIM, EM_DIM, methods):
    if not func_name in ['Ackley', 'Levy', 'Griewank']:
        raise NotImplementedError

    res_p = Path(f"results/{func_name}-full/")

    ## HDBO
    if 'saasbo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-saasbo.npy")), 'SAASBO')
    if 'rembo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-rembo.npy")), "REMBO")
    if 'alebo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-alebo.npy")), "ALEBO")
    if 'turbo1' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turbo1.npy")), 'TuRBO')
    if 'turbom' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turbom-1.npy")), 'TuRBOM-1')
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turbom.npy")), 'TuRBOM')
    if 'gibo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-gibo.npy")), 'GIBO')
    if 'mpd' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-mpd.npy")), 'BO-MPD')
    if 'turbo-d' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turboD.npy")), 'TuRBO-D')
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turboD-nw.npy")), 'TuRBO-D-nW')
    if 'turbo-d-drop' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turboD-drop.npy")), 'TuRBO-D')
        # plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-turboD-drop-nw.npy")), 'TuRBO-D-nW')
    if 'gp-ei' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-bo.npy")), "GP-EI")
    if 'hebo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-hebo.npy")), label="HEBO")
    if 'bo_moo' in methods:
        plot_data(data=np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-bo_moo.npy")), label="HEBO")
    if 'sobol' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-sobol.npy")), "Sobol")
    if 'sir-bo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-sir_bo.npy")), "SIR-BO")
    if 'ksir-bo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-ksir_bo.npy")), "KSIR-BO")
    if 'kdr-bo' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-kdr_bo.npy")), "KDR-BO")
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-d{EM_DIM}-mkdr_bo.npy")),
    #     label="MKDR-BO"
    # )
    if 'add-gp-ucb' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-add_bo.npy")), "Add-GP-UCB")
    if 'cmaes' in methods:
        plot_data(np.load(res_p.joinpath(f"{func_name}-full-D{DIM}-cmaes.npy")), "CMA-ES")
    ax.grid(True)
    ax.set_title(f"{func_name} (D = {DIM})", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    # ax.set_ylim([0, 600])
    ax.legend()
    # plt.savefig(fig_p)
    plt.show()
# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-mkdr_bo2.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="MKDR-BO2")


# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-full-D{DIM}-hebo.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="HEBO")
# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-full-D{DIM}-d{EM_DIM}-manifold_bo.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="MBO")
# ax.plot([0, len(store_data)], [func.optimal_value, func.optimal_value], "k--", lw=3)


if __name__ == '__main__':
    dim = 100
    methods=[
        # 'saasbo',
        # 'alebo',
        # 'rembo',
        # 'sir-bo',
        # 'ksir-bo',
        # 'add-gp-ucb', 
        # 'turbo1',
        # 'turbom',
        # 'gibo',
        # 'mpd',
        # 'turbo-d',
        # 'turbo-d-drop',
        # 'bo_moo',
        # 'sobol',
        'cmaes'
    ]
    plotFunc_f(func_name='Ackley', DIM=dim, EM_DIM=10, methods=methods)
    # plotFunc_f(func_name='Levy', DIM=dim, EM_DIM=10, methods=methods)
    # plotFunc_f(func_name='Griewank', DIM=dim, EM_DIM=10, methods=methods)
