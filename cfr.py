# CFR implementation for Kuhn Poker
# Following Neller and Lanctot. An Introduction to Conterfactual Regret Minimization.
# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
#
# Kuhn Poker action sequence
# P1 b/p |  P2 b/p | P1 b/p (if not p,p)
#
# Information sets
# Pre-bet deal
# 1) P1 1, P2 2/3
# 2) P1 2, P2 1/3
# 3) P1 3, P2 1/2
# 
# Non-terminal histories: H\Z = {'', 'P', 'B', 'PB'}
#  => |H\Z| = 4
# 3 cards x 4 bet sub-sequences  => 12 infosets

import numpy as np
np.set_printoptions(suppress=True)

ACTIONS = ['P', 'B']
NUM_ACTIONS = len(ACTIONS)
NUM_CARDS = 3

def normalize(strategy):
    ss = np.sum(strategy)
    if ss > 0:
        return strategy / ss
    else:
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

def sample_action(strategy):
    assert np.abs(np.sum(strategy) - 1.0) < 1e-9
    return np.searchsorted(np.cumsum(strategy), np.random.rand())

class Node:
    def __init__(self, infoset):
        self.infoset = infoset
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    def get_strategy(self):
        return normalize(np.maximum(self.regret_sum, 0))

    def get_avg_strategy(self):
        return normalize(self.strategy_sum)

    def __str__(self):
        return f'sigma({self.infoset}): {self.get_avg_strategy()}'

def get_payoffs(cards, history):

    if len(history) < 2 or history == 'PB':
        # Non-terminal state
        return None
    if history == 'BP':
        # Player 2 folds
        return +1, -1
    if history == 'PBP':
        # Player 1 folds
        return -1, +1

    # Blind + bet for each player
    assert history.count('B') % 2 == 0, 'Bets must match in non-fold terminal states'
    pot_size = 1 + history.count('B') / 2
    assert pot_size in [1, 2], 'Pots may contain at most 2 chips'

    # Award pot to winner
    assert cards[0] != cards[1], 'Game can never end in draw.'
    if cards[0] > cards[1]:
        return +pot_size, -pot_size
    else:
        return -pot_size, +pot_size

def infoset_name(card, history):
    return f'{card}{history}'

def cfr(cards, history, p0, p1, node_map):
    player = len(history) % 2

    # If terminal state, return payoff
    TERMINAL_HISTORIES = ['BB', 'BP', 'PP', 'PBP', 'PBB']
    payoffs = get_payoffs(cards, history)
    if payoffs is not None:
        assert history in TERMINAL_HISTORIES, f'Unexpected terminal history: {history}'
        return payoffs[player]
    assert history not in TERMINAL_HISTORIES

    infoset = infoset_name(cards[player], history)

    # get/create infoset node
    node = node_map.get(infoset, Node(infoset))
    node_map[infoset] = node

    # for each action, recursively invoke cfr
    strategy = node.get_strategy()
    utility = np.zeros(NUM_ACTIONS)
    for i, action in enumerate(ACTIONS):
        p0n = p0 * (strategy[i] if player == 0 else 1)
        p1n = p1 * (strategy[i] if player == 1 else 1)
        utility[i] = -cfr(cards, history + action, p0n, p1n, node_map)
    node_utility = np.sum(strategy * utility)

    # update node strategy statistics
    realization_weight = p0 if player == 0 else p1
    node.strategy_sum += realization_weight * strategy

    # update node regret statistics
    regret = utility - node_utility
    realization_weight = p1 if player == 0 else p0
    node.regret_sum += realization_weight * regret

    return node_utility

def train(niter):
    node_map = {}
    util = 0
    cards = list(range(NUM_CARDS))
    for _ in range(niter):
        np.random.shuffle(cards)
        util += cfr(cards, "", 1, 1, node_map)
    print('Finished training with an avg utility of', (util / niter))
    return node_map

def _avg_util(niter):
    total_util = 0
    cards = list(range(NUM_CARDS))
    for _ in range(niter):
        np.random.shuffle(cards)
        history = ''
        terminal_game_utility = None
        while terminal_game_utility is None:
            player = len(history) % 2
            infoset = infoset_name(cards[player], history)
            strategy = node_map[infoset].get_avg_strategy()
            a = sample_action(strategy)
            history += ACTIONS[a]
            terminal_game_utility = get_payoffs(cards, history)
        total_util += terminal_game_utility[0]
    return total_util / niter

def evaluate_utility(nouter, ninner, node_map):
    import multiprocessing as mp
    utilities = [0] * nouter
    pool = mp.Pool()
    for i, util in enumerate(pool.imap_unordered(_avg_util, [ninner] * nouter)):
        utilities[i] = util
    pool.close()
    print(f'Completed {nouter} trials of {ninner} games.')
    print(f'  Mean utility: {np.mean(utilities):0.5f}')
    print(f'  Std: {np.std(utilities):0.5f}')

def KL(p, q):
    Hp = -np.sum(p[p > 0] * np.log(p[p > 0]))
    Hpq = -np.sum(p[p > 0] * np.log(q[p > 0]))
    return Hpq - Hp

def evaluate_optimality(node_map):
    p0_jack_strategy = node_map[infoset_name(0, '')].get_avg_strategy()
    alpha = p0_jack_strategy[ACTIONS.index('B')]
    if alpha > 1/3:
        print('WARNING: Nash-equilibrium alpha not in valid domain:', a, 'not in [0, 1/3]')
    
    # Equilibrium strategy solved by Kuhn, as described at https://en.wikipedia.org/wiki/Kuhn_poker
    NASH_EQ = [
        # Player 1 strategy
        # (Card seen by current player, bet history, p_bet)
        (0, '', alpha),
        (1, '', 0.0),
        (2, '', 3  * alpha),

        # Player 2 strategy
        (0, 'B', 0.0),
        (0, 'P', 1 / 3),
        (1, 'B', 1 / 3),
        (1, 'P', 0.0),
        (2, 'B', 1.0),
        (2, 'P', 1.0),

        # Player 1 strategy
        (0, 'PB', 0.0),
        (1, 'PB', alpha + 1 / 3),
        (2, 'PB', 1.0),
    ]

    def peq(p_bet):
        strategy_eq = np.zeros(NUM_ACTIONS)
        strategy_eq[ACTIONS.index('P')] = 1 - p_bet
        strategy_eq[ACTIONS.index('B')] = p_bet
        return strategy_eq

    print('KL(sigma* | sigma)')
    for card, history, p_bet in NASH_EQ:
        infoset = infoset_name(card, history)
        node = node_map[infoset]
        kl = KL(peq(p_bet), node.get_avg_strategy())
        print(f'  {node}: KL={kl:0.5f}')


if __name__ == '__main__':
    node_map = train(1000000)
    print('------------')
    evaluate_utility(10, 100000, node_map)
    print('------------')
    evaluate_optimality(node_map)
