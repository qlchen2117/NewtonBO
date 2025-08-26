# Mujoco
from testReal.testRL import testRL
methods = [
    # 'saasbo',
    # 'alebo',
    # 'rembo',
    # 'sir-bo',
    # 'ksir-bo'
    # 'turbo1',
    # 'turbom',
    # 'gibo',
    # 'bo-mpd',
    # 'turbo-d',
    # 'turbo-d-drop',
    # 'add-gp-ucb',
    # 'hebo',
    # 'gp-ei',
    # 'gp-pes',
    # 'sobol'
    'cmaes'
]

benchmarks = [
    'Hopper',
    'HalfCheetah',
    'Walker2d',
]

N_ITERACTIONS, BATCH_SIZE, N_INIT = 100, 5, 50
# N_ITERACTIONS, BATCH_SIZE, N_INIT = 80, 5, 50

for func_name in benchmarks:
    testRL(func_name, N_EPOCH=20, N_ITERACTIONS=N_ITERACTIONS, N_INIT=N_INIT, BATCH_SIZE=BATCH_SIZE, methods=methods)

# from testReal.plotReal import plotReal

# plotReal('Hopper', 10, methods)