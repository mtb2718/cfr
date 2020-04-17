
import numpy as np
np.set_printoptions(suppress=True)


ACTION_NAMES = ['R', 'P', 'S']
NUM_ACTIONS = len(ACTION_NAMES)


def get_strategy(regret_sum):
    return normalize(np.maximum(regret_sum, 0))

def sample_action(strategy):
    assert np.abs(np.sum(strategy) - 1.0) < 1e-9
    return np.searchsorted(np.cumsum(strategy), np.random.rand())

def normalize(strategy):
    ss = np.sum(strategy)
    if ss > 0:
        return strategy / ss
    else:
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

def train(niter):
    opponent_strategy = np.array([0.4, 0.3, 0.3])
    strategy_sum = np.zeros(NUM_ACTIONS)
    regret_sum = np.zeros(NUM_ACTIONS)

    for _ in range(niter):
        strategy = get_strategy(regret_sum)
        strategy_sum += strategy
        action = sample_action(strategy)
        opponent_action = sample_action(opponent_strategy)
        
        utility = np.zeros(NUM_ACTIONS)
        utility[opponent_action] = 0
        utility[(opponent_action + 1) % NUM_ACTIONS] = +1
        utility[(opponent_action - 1) % NUM_ACTIONS] = -1

        regret = utility - utility[action]
        regret_sum += regret

    return normalize(strategy_sum)

def train2(niter):
    strategy_sum0 = np.zeros(NUM_ACTIONS)
    strategy_sum1 = np.zeros(NUM_ACTIONS)
    regret_sum0 = np.zeros(NUM_ACTIONS)
    regret_sum1 = np.zeros(NUM_ACTIONS)

    for _ in range(niter):
        strategy0 = get_strategy(regret_sum0)
        strategy1 = get_strategy(regret_sum1)
        strategy_sum0 += strategy0
        strategy_sum1 += strategy1

        action0 = sample_action(strategy0)
        action1 = sample_action(strategy1)

        utility0 = np.zeros(NUM_ACTIONS)
        utility0[action1] = 0
        utility0[(action1 + 1) % NUM_ACTIONS] = +1
        utility0[(action1 - 1) % NUM_ACTIONS] = -1

        utility1 = np.zeros(NUM_ACTIONS)
        utility1[action0] = 0
        utility1[(action0 + 1) % NUM_ACTIONS] = +1
        utility1[(action0 - 1) % NUM_ACTIONS] = -1

        regret_sum0 += utility0 - utility0[action0]
        regret_sum1 += utility1 - utility1[action1]

    return normalize(strategy_sum0), normalize(strategy_sum1)


if __name__ == '__main__':
    strategies = train2(100000)
    print('Final strategies:')
    print(strategies[0])
    print(strategies[1])
