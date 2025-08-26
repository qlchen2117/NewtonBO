# Mujoco
from testReal.testRover import testRover
methods = [
    # 'saasbo',
    # 'alebo',
    # 'rembo',
    # 'sir-bo',
    # 'ksir-bo',
    # 'turbo1',
    # 'turbom',
    # 'gibo',
    # 'bo-mpd',
    # 'turbo-d',
    # 'turbo-d-drop',
    # 'add-gp-ucb',
    # 'hebo',
    # 'gp-ei',
    # 'sobol',
    'cmaes'
]

N_ITERACTIONS, BATCH_SIZE, N_INIT = 100, 5, 50
testRover(N_EPOCH=20, N_ITERACTIONS=N_ITERACTIONS, N_INIT=N_INIT, BATCH_SIZE=BATCH_SIZE, methods=methods)

# from testReal.plotReal import plotReal

# plotReal('Rover', 10, methods)