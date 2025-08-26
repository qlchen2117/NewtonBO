import numpy as np
from testFunctions.testFull import testFull

N_EPOCH = 20
DIM, EM_DIM, N_ITERACTIONS, BATCH_SIZE, N_INIT, MAX_TIME = 50, 10, 20, 5, 50, np.inf
# DIM, EM_DIM, N_ITERACTIONS, BATCH_SIZE, N_INIT, MAX_TIME = 100, 10, 40, 5, 50, np.inf
# DIM, EM_DIM, N_ITERACTIONS, BATCH_SIZE, N_INIT, MAX_TIME = 50, 10, float("inf"), 5, 50, 300
# DIM, EM_DIM, N_ITERACTIONS, BATCH_SIZE, N_INIT, MAX_TIME = 100, 10, float("inf"), 5, 50, 300

objectives = [
    'Ackley',
    # 'Levy',
    'Griewank',
]

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
    # 'gp-pes',
    # 'sobol',
    'cmaes',
    # 'newton-bo-rand',
]

for obj in objectives:
    testFull(obj, N_EPOCH=N_EPOCH, DIM=DIM, EM_DIM=EM_DIM, N_ITERACTIONS=N_ITERACTIONS, MAX_TIME=MAX_TIME, N_INIT=N_INIT, BATCH_SIZE=BATCH_SIZE, methods=methods)


# # Linear Subspace
# from testFunctions.testLinsub import testLinsub
# testLinsub('Ackley', N_EPOCH=20)
# testLinsub('Levy', N_EPOCH=10)
# testLinsub('Griewank', N_EPOCH=20)

# Nonlinear Subspace
# from testFunctions.testNlinsub import testNlinsub
# DIM = 100
# testNlinsub(func_name='Ackley', N_EPOCH=3, DIM=DIM)
# testNlinsub(func_name='Levy', N_EPOCH=3, DIM=DIM)
# testNlinsub(func_name='Griewank', N_EPOCH=3, DIM=DIM)

# Plot
# from testFunctions.plotLinsub import plotNlinsub
# plotNlinsub(func_name='Ackley')
