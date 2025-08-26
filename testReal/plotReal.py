import numpy as np
from pathlib import Path
# Draw pictures
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

def plotReal(bench_name, EM_DIM, methods, n_iter=np.inf):
    def plot_data(data, label, maximize=False, n_iter=np.inf):
        print(f"{label}: {data.shape}")
        if maximize:
            f = np.maximum.accumulate(data * -1, axis=-1)
        else:
            f = np.minimum.accumulate(data, axis=-1)
        n_iter = min(n_iter, data.shape[-1])
        x = np.arange(1, n_iter+1)
        mean = f.mean(0)
        sem = f.std(0) / (f.shape[0] ** 0.5)
        ax.plot(x, mean[:n_iter])
        ax.fill_between(x, (mean - 2. * sem)[:n_iter], (mean + 2. * sem)[:n_iter], alpha = 0.3, label=label)

    if bench_name == 'nas':
        DIM=36
        is_maximize = True
        res_p = Path(f"results/{bench_name}/")
        ax.set_ylim([0.91, 0.94])
    elif bench_name == 'DNA':
        DIM = 180
        is_maximize = False
        res_p = Path(f"results/lassobench/final")
    elif bench_name == 'synt_high':
        DIM = 300
        is_maximize = False
        res_p = Path(f"results/lassobench/final")
    elif bench_name == 'mopta':
        DIM = 124
        is_maximize = False
        res_p = Path(f"results/mopta/final")
    elif bench_name == 'Hopper':
        DIM, is_maximize, res_p = 33, True, Path("results/mujoco/final")
    elif bench_name == 'Walker2d' or bench_name == 'HalfCheetah':
        DIM, is_maximize, res_p = 102, True, Path("results/mujoco")
    elif bench_name == 'Push':
        DIM, is_maximize, res_p = 14, True, Path("results/push")
    elif bench_name == 'Rover':
        DIM, is_maximize, res_p = 60, True, Path("results/rover/")
    else:
        raise NotImplementedError
    # Read results
    ## HDBO
    if 'saasbo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-saasbo.npy")), "SAASBO", is_maximize)
    if 'turbo1' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turbo1.npy")), "TuRBO", is_maximize)
    if 'turbom' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turbom.npy")), "TuRBOM", is_maximize)
        # plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turbom-1.npy")), "TuRBOM-1", is_maximize)
    if 'gibo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-gibo.npy")), 'GIBO')
    if 'turbo-d' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turboD.npy")), "TuRBO-D", is_maximize)
    if 'turbo-d-drop' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turboD-drop.npy")), "TuRBO-D", is_maximize)
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turboD-drop-nw.npy")), "TuRBO-D-nW", is_maximize)
    if 'rembo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-rembo.npy")), "REMBO", is_maximize)
    if 'alebo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-alebo.npy")), "ALEBO", is_maximize)
    if 'add-gp-ucb' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-add_bo.npy")), "Add-GP-UCB", is_maximize)
    if 'sir-bo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-sir_bo.npy")), "SIR-BO", is_maximize)
    if 'ksir-bo' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-ksir_bo.npy")), "KSIR-BO", is_maximize)
    if 'sobol' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-sobol.npy")), "Sobol", is_maximize)
    if 'gp-ei' in methods:
        plot_data(np.load(res_p.joinpath(f"{bench_name}-D{DIM}-gp-ei.npy")), "GP-EI", is_maximize)
    # plot_data(
    #     np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-kdr_bo.npy")),
    #     "KDR-BO", is_maximize
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-mkdr_bo.npy")),
    #     "MKDR-BO", is_maximize
    # )

    ax.grid(True)
    ax.set_title(f"{bench_name} (D = {DIM})", fontsize=18)
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_ylabel("Test accuracy", fontsize=18)
    # ax.set_ylim([0, 100])
    ax.legend()
    plt.show()
    # plt.savefig(f"{bench_name}.pdf")

## BO
# Y_hebo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-hebo.npy")), axis=0)
# Y_bo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo.npy")), axis=0)
# Y_bo_warp = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo_warp.npy")), axis=0)
# Y_bo_moo = np.mean(np.load(res_p.joinpath(f"Top2-D{DIM}-moo.npy")), axis=0)
# Y_sobol = np.mean(np.load(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-sobol.npy"), axis=0)


# HDBO
# ax.plot(np.maximum.accumulate(Y_sir_bo), label="SIR-BO")

# BO
# ax.plot(np.maximum.accumulate(Y_hebo), label="HEBO")
# ax.plot(np.maximum.accumulate(Y_bo), label="BO")
# ax.plot(np.maximum.accumulate(Y_bo_warp), label="BO-Warp")
# ax.plot(np.maximum.accumulate(Y_bo_moo), label="BO-MOO")
# ax.plot(np.maximum.accumulate(Y_sobol), label="SOBOL")


# ax.plot([0, len(store_data)], [1, 1], "k--", lw=3)
if __name__ == '__main__':
    methods=[
        'saasbo',
        'alebo',
        'add-gp-ucb',
        'sir-bo',
        'ksir-bo',
        # 'turbo1',
        'turbom',
        'gibo',
        # 'mpd',
        # 'turbo-d',
        'turbo-d-drop',
        # 'bo_moo',
        'sobol'
    ]
    # plotReal('nas', 10, methods)
    # plotReal('DNA', 10, methods)
    # plotReal('mopta', 10, methods)
    plotReal('synt_high', 10, methods)