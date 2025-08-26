# LassoBench
from testReal.testLasso import testLasso

methods = [
    # 'saasbo'
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
    # 'sobol',
    # 'add-gp-ucb',
    # 'hebo',
    # 'gp-ei',
    # 'sobol',
    'cmaes'
]

benchmarks = [
#    'DNA',
    'synt_high'
]

N_ITERACTIONS, BATCH_SIZE, N_INIT = 40, 5, 50

for pick_data in benchmarks:
    testLasso(pick_data, N_EPOCH=3, N_ITERACTIONS=N_ITERACTIONS, N_INIT=N_INIT, BATCH_SIZE=BATCH_SIZE, methods=methods)
