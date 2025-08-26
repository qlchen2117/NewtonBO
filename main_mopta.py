# Mopta
from testReal.testMopta import testMopta
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
    # 'sobol',
    # 'add-gp-ucb',
    # 'hebo',
    # 'gp-ei',
    # 'sobol'
    'cmaes'
]

N_ITERACTIONS, BATCH_SIZE, N_INIT = 40, 5, 50

testMopta(N_EPOCH=20, N_ITERACTIONS=N_ITERACTIONS, N_INIT=N_INIT, BATCH_SIZE=BATCH_SIZE, methods=methods)
