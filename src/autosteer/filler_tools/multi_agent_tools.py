# Built-in packages
import hashlib
import json
import os
import random
import re
from itertools import product
from string import ascii_uppercase
# External packages
import numpy as np
import pandas as pd
# Local packages
from .fill_registry import FillRegister
function_registry = FillRegister()
ALPHABET = list(ascii_uppercase)

class GameMatrix():
    def __init__(self, n=2, m=2, payoff_type=int, payoff_range=(0, 100), game_type = None, row_action_labels = None, column_action_labels = None, decimals = 2):
        self.n = n
        self.m = m
        self.payoff_type = payoff_type
        self.payoff_range = payoff_range
        self.game_type = game_type
        self.decimals = decimals
        self.payoffs = self.generate_payoff_matrix()
        self.row_action_labels = row_action_labels
        self.column_action_labels = column_action_labels

    def generate_payoff_matrix(self):
        if isinstance(self.payoff_type, int):
            n_matrix = np.random.randint(*self.payoff_range, size=(self.n, self.m))
            m_matrix = np.random.randint(*self.payoff_range, size=(self.n, self.m))
        #elif isinstance(self.payoff_type, float):
        else:
            n_matrix = np.random.rand(self.n, self.m) * abs(self.payoff_range[0] - self.payoff_range[1]) + self.payoff_range[0]
            m_matrix = np.random.rand(self.n, self.m) * abs(self.payoff_range[0] - self.payoff_range[1]) + self.payoff_range[0]
            n_matrix = n_matrix.round(self.decimals)
            m_matrix = m_matrix.round(self.decimals)
        
        payoff_matrix = [[(n_matrix[i][j], m_matrix[i][j]) for j in range(self.m)] for i in range(self.n)]
        
        if self.game_type == 'symmetric':
            assert self.n == self.m, "Symmetric games must have the same number of rows and columns."
            m1 = (n_matrix + n_matrix.T)/2
            m2 = (m_matrix + m_matrix.T)/2
            payoff_matrix = [[(m1[i][j], m2[i][j]) if i < j else (m2[i][j], m1[i][j]) for j in range(self.n)] for i in range(self.n)]

            for i in range(self.n):
                diag = np.random.choice([payoff_matrix[i][i][0], payoff_matrix[i][i][1]])
                payoff_matrix[i][i] = (diag, diag)
            return payoff_matrix
        return payoff_matrix

    def best_response(self, player, action):
        """
        Returns the best response for a player given an action of the opponent player.
        """
        assert player in [0, 1], "Player must be either 0: row or 1: column."
        assert action in range(self.m) if player == 0 else action in range(self.n), "Action must be a valid index for the opponent player."

        if player == 0:
            return np.argmax([self.payoffs[i][action][0] for i in range(self.n)])
        return np.argmax([self.payoffs[action][j][1] for j in range(self.m)])
    
    def compute_minmax(self, player):
        assert player in [0, 1], "Player must be either 0: row or 1: column."
        if player == 0:
            return min(self.payoffs[self.best_response(player, i)][i][0] for i in range(self.m))
        return min(self.payoffs[i][self.best_response(player, i)][1] for i in range(self.n))
    
    def flatten_matrix(self):
        return [payoff for row in self.payoffs for payoff_tuple in row for payoff in payoff_tuple]

    def get_min_payoff(self):
        return min(self.flatten_matrix())
    
    def get_max_payoff(self):
        return max(self.flatten_matrix())

    def __str__(self):
        if self.row_action_labels is not None and self.column_action_labels is not None:
            return pd.DataFrame(self.payoffs, index=[f"Action {i}" for i in self.row_action_labels], columns=[f"Action {i}" for i in self.column_action_labels]).to_string()
        return pd.DataFrame(self.payoffs, index=[f"Action {i}" for i in range(self.n)], columns=[f"Action {i}" for i in range(self.m)]).to_string()

        

class DecisionNode():
    def __init__(self, player, parent, round_num):
        self.left = None
        self.right = None
        self.player = player
        self.parent = parent
        self.round_num = round_num
        self.opt_player_decision = None
        self.opt_player_payoff = 0
        self.value = None
    
    def get_payoffs(self, player):
        return self.opt_player_payoff

    def __str__(self):
        return f"Round: {self.round_num}, Player: {self.player}, Decision: {self.opt_player_decision}, Payoff: {self.opt_player_payoff}"

class PayoffNode():
    def __init__(self, payoff_type = int, payoff_range = (-10, 10), num_players=2):
        # payoff_type is the type itself (int or float); the old isinstance() checks never matched it.
        if payoff_type is int:
            self.payoffs = np.random.randint(payoff_range[0], payoff_range[1], num_players)
        elif payoff_type is float:
            # rounded to cents so the payoffs shown in the question are the ones solved
            self.payoffs = np.round(np.random.uniform(payoff_range[0], payoff_range[1], num_players), 2)
        else:
            raise ValueError("Invalid payoff type")

    def get_payoffs(self, player):
        """Payoff of player 1 or 2 (1-based)."""
        return self.payoffs[player - 1]

    def get_payoff_vector(self):
        return self.payoffs

    def __str__(self):
        return str(self.payoffs)

class GameTree():
    def __init__(self, num_rounds, num_players = 2, payoff_type = int, payoff_range = (-10, 10)):
        self.root = None
        self.num_rounds = num_rounds
        self.num_players = num_players
        self.payoff_type = payoff_type
        self.payoff_range = payoff_range
        assert num_players <= num_rounds, "Number of players must be less than or equal to the number of rounds"
        assert num_rounds <= 3, "Number of rounds must be less than or equal to 3"
        self.root = self.generate_tree()
    
    def generate_tree(self):
        decision_nodes = [DecisionNode(player = i % self.num_players + 1, parent = None, round_num = i) for i in range(self.num_rounds)]
        for i, node in enumerate(decision_nodes):
            node.parent = decision_nodes[i-1] if i > 0 else None 
        payoff_nodes = [PayoffNode(payoff_type = self.payoff_type, payoff_range = self.payoff_range, num_players = self.num_players) for _ in range(self.num_rounds + 1)]
        for i, node in enumerate(decision_nodes[:-1]):
            node.left = payoff_nodes[i]
            node.right = decision_nodes[i+1]
        decision_nodes[-1].left = payoff_nodes[-2]
        decision_nodes[-1].right = payoff_nodes[-1]
        return decision_nodes[0]


    def get_root(self):
        self.root

    def get_decision_node(self, round_num):
        node = self.root
        for _ in range(round_num):
            node = node.right
        return node
    
    def get_last_decision_node(self):
        return self.get_decision_node(self.num_rounds - 1)

    def solve(self):
        """
        Backward induction over every decision node, from the last to the first. At each node the mover
        (node.player) compares their own payoff from ending ('left', the payoff node) with their own payoff
        from the continuation ('right', the value of the next node). Returns False if any mover is
        indifferent (the equilibrium path would be ambiguous), else True.
        """
        node = self.get_last_decision_node()
        no_ties = True
        while node is not None:
            end = node.left.get_payoff_vector()
            cont = node.right.get_payoff_vector() if isinstance(node.right, PayoffNode) else node.right.value
            u_end, u_cont = end[node.player - 1], cont[node.player - 1]
            if u_end == u_cont:
                no_ties = False
            node.opt_player_decision = 'left' if u_end > u_cont else 'right'
            node.value = end if node.opt_player_decision == 'left' else cont
            node.opt_player_payoff = node.value[node.player - 1]
            node = node.parent
        return no_ties

    def compute_backwards_induction(self, round_num, player=None):
        """
        Subgame-perfect choice at decision round `round_num` (1-based: round 1 is the root, moved by
        player 1; round 2 by player 2; round 3 by player 1): 'left' (end) or 'right' (continue).
        Returns None if any node has a tie. `player` is accepted for backward compatibility; the mover
        is always the node's own player.
        """
        if not self.solve():
            return None
        return self.get_decision_node(round_num - 1).opt_player_decision


def generate_game_matrix(n, m, payoff_type, action_names, payoff_min = 0, payoff_max = 20):
    # Generate random payoff matrices for players
    if payoff_type == 'integer':
        row_player_df = np.random.randint(payoff_min, payoff_max, size=(n, m))
        col_player_df = np.random.randint(payoff_min, payoff_max, size=(n, m))
    else:
        row_player_df = (np.random.rand(n, m) * (abs(payoff_min) + abs(payoff_max)) - payoff_min).round(2)
        col_player_df = (np.random.rand(n, m) * (abs(payoff_min) + abs(payoff_max)) - payoff_min).round(2)
    

    # Create dataframes for display
    row_player_df = pd.DataFrame(row_player_df, index=[action_names['row'][i] for i in range(n)], columns=[action_names['column'][i] for i in range(m)])
    col_player_df = pd.DataFrame(col_player_df, index=[action_names['row'][i] for i in range(n)], columns=[action_names['column'][i] for i in range(m)])

    return row_player_df, col_player_df

def get_bayesian_games(game, prob_dist, display_type = 'single_out'):
    '''
    game is a tuple with the format (row_player_df, col_player_df) where row_player_df and col_player_df are dataframes
    prob_dist is a list of probabilities that sum to 1
    returns len(prob_dist) number of games whose expected value is equal to game
    '''
    row_player_df, col_player_df = game
    bayesian_games = []
    for prob in prob_dist:
        bayesian_row_player_df = round(row_player_df * prob, 2)
        bayesian_col_player_df = round(col_player_df * prob, 2)
        if display_type == 'single_out':
            bayesian_games.append(display_game_matrix(bayesian_row_player_df, bayesian_col_player_df))
        elif display_type == 'double_out':
            bayesian_games.append((bayesian_row_player_df, bayesian_col_player_df))
    return bayesian_games


def display_game_matrix(row_player_df, col_player_df):
    # Combine the two dataframes to create a single dataframe with pairs of payoffs
    df_combined = pd.DataFrame(index=row_player_df.index, columns=col_player_df.columns)

    for i in range(row_player_df.shape[0]):
        for j in range(row_player_df.shape[1]):
            df_combined.iloc[i, j] = f'({row_player_df.iloc[i, j]}, {col_player_df.iloc[i, j]})'

    return df_combined

def find_nash_equilibrium(row_player_df, col_player_df):
    # Check for Nash equilibrium
    equilibrium_indices = []

    for row in range(row_player_df.shape[0]):
        for col in range(row_player_df.shape[1]):
            row_value = row_player_df.iloc[row, col]
            col_value = col_player_df.iloc[row, col]

            # Check if (row_value, col_value) is a Nash equilibrium
            if (
                row_value == max(row_player_df.iloc[:, col]) and
                col_value == max(col_player_df.iloc[row, :])
            ):

                equilibrium_indices.append((row, col))

    return equilibrium_indices if equilibrium_indices else ()

def find_dom_strategies(row_player_df, col_player_df, player_type, dom_type):
    if player_type == 'row':
        df = row_player_df
        dom_rows = {}

        for i, row1 in df.iterrows():
            dom_rows[i] = []

            for j, row2 in df.iterrows():
                if dom_type == 'dominant':
                    if all(row1 > row2):
                        dom_rows[i].append(j)
                elif dom_type == 'dominated':
                    if all(row1 < row2):
                        dom_rows[i].append(j)
        return dom_rows
    elif player_type == 'column':
        df = col_player_df
        dom_columns = {}

        for col1 in df.columns:
            dom_columns[col1] = []

            for col2 in df.columns:
                if dom_type == 'dominant':
                    if all(df[col1] > df[col2]):
                        dom_columns[col1].append(col2)
                elif dom_type == 'dominated':
                    if all(df[col1] < df[col2]):
                        dom_columns[col1].append(col2)

        return dom_columns

def gen_iterations(game_size, num_iterations, player_type):
    type2index = [{'row': 1, 'column': 0}, {'row': 0, 'column': 1}]
    return flatten_list([np.eye(game_size[type2index[iteration%2][player_type]]-iteration//3, dtype=int)[np.random.choice(game_size[type2index[iteration%2][player_type]]-iteration//3, 1)].tolist() for iteration in range(1, num_iterations+1)])

def flatten_list(lst):
    return [item for elm in lst for item in elm]

'''
game_size is a tuple

solution is a dictionary with formats:
    1) {'nash': int}
    2) {'dom': tuple, 'player_type': 'row'/'column', 'dom_type': 'dominant'/'dominated'}
    3) {'iterated': tuple, 'player_type': 'row'/'column'}
'''
def generate_game(game_size, solution, action_names = {'row': ['Action R', 'Action S', 'Action T', 'Action U'], 'column': ['Action K', 'Action L', 'Action M', 'Action N']}):
    '''
    returns the index of the correct answer
    '''
    assert solution.get('nash', 0) <= np.prod(game_size), f"Can only have at most {np.prod(game_size)} number of pure Nash for a {game_size[0]}x{game_size[1]} size game."

    get_other = {'row': 'column', 'column': 'row', 'dominant': 'dominated', 'dominated': 'dominant'}

    if 'nash' in solution:
        row_player_df, col_player_df = generate_game_matrix(*game_size, solution.get('payoff_type'), action_names)
        equilibrium_indices = find_nash_equilibrium(row_player_df, col_player_df)
        while len(equilibrium_indices) != solution.get('nash'):
            row_player_df, col_player_df = generate_game_matrix(*game_size, solution.get('payoff_type'), action_names)
            equilibrium_indices = find_nash_equilibrium(row_player_df, col_player_df)
        df_combined = display_game_matrix(row_player_df, col_player_df)

        equilibrium_indices = [(list(df_combined.index)[eq_idx[0]], list(df_combined.columns)[eq_idx[1]]) for eq_idx in equilibrium_indices]

        if solution.get('answer_type') == 'both':
            options = list(product(list(df_combined.index), list(df_combined.columns)))
            if solution.get('display_type') == 'double_out':
                return row_player_df, col_player_df, options, [options.index(eq_idx) for eq_idx in equilibrium_indices]
            return df_combined, options, [options.index(eq_idx) for eq_idx in equilibrium_indices]
        elif solution.get('answer_type') == 'row':
            options = list(df_combined.index)
            if solution.get('display_type') == 'double_out':
                return row_player_df, col_player_df, options, [options.index(eq_idx[0]) for eq_idx in equilibrium_indices]
            return df_combined, options, [options.index(eq_idx[0]) for eq_idx in equilibrium_indices]
        options = list(df_combined.columns)
        if solution.get('display_type') == 'double_out':
            return row_player_df, col_player_df, options, [options.index(eq_idx[1]) for eq_idx in equilibrium_indices]
        return df_combined, options, [options.index(eq_idx[1]) for eq_idx in equilibrium_indices]


    elif 'dom' in solution:
        row_player_df, col_player_df = generate_game_matrix(*game_size, solution.get('payoff_type'), action_names)
        dom_indices = find_dom_strategies(row_player_df, col_player_df, solution.get('player_type'), solution.get('dom_type'))

        while [len(dom_indices[action_name]) for action_name in random.sample(list(dom_indices.keys()), len(dom_indices.keys()))] != solution.get('dom'):
            row_player_df, col_player_df = generate_game_matrix(*game_size, solution.get('payoff_type'), action_names)
            dom_indices = find_dom_strategies(row_player_df, col_player_df, solution.get('player_type'), solution.get('dom_type'))

        if solution.get('answer_type') == 'bimatrix':
            df_combined = display_game_matrix(row_player_df, col_player_df)
        else:
            if solution.get('player_type') == 'row':
                df_combined = row_player_df
            else:
                df_combined = col_player_df

        return df_combined, list(dom_indices.keys()), max(dom_indices, key = lambda x: len(set(dom_indices[x])))
    
    elif 'iterated' in solution:
        player_axis = {'row': 0, 'column': 1}

        iterated_indices = []
        iterated_actions = []

        while iterated_indices != solution.get('iterated'):
            iterated_indices = []
            iterated_actions = []
            row_player_df, col_player_df = generate_game_matrix(*game_size, solution.get('payoff_type'), action_names)
            iterated_dfs = [display_game_matrix(row_player_df, col_player_df)]
            temp_row_player_df, temp_col_player_df = row_player_df.copy(), col_player_df.copy()
            for iteration, num in enumerate(solution.get('iterated')):
                if iteration % 2 == 0:
                    dom_indices = find_dom_strategies(temp_row_player_df, temp_col_player_df, solution.get('player_type'), 'dominated')
                    iterated_indices.append([len(dom_indices[action_name]) for action_name in dom_indices])
                    iterated_actions.append(dom_indices)
                    temp_row_player_df = temp_row_player_df.drop([dom_action for dom_action in dom_indices if dom_indices[dom_action]], axis = player_axis[solution.get('player_type')])
                    temp_col_player_df = temp_col_player_df.drop([dom_action for dom_action in dom_indices if dom_indices[dom_action]], axis = player_axis[solution.get('player_type')])
                    iterated_dfs.append(display_game_matrix(temp_row_player_df, temp_col_player_df))
                else:
                    dom_indices = find_dom_strategies(temp_row_player_df, temp_col_player_df, get_other[solution.get('player_type')], 'dominated')
                    iterated_indices.append([len(dom_indices[action_name]) for action_name in dom_indices])
                    iterated_actions.append(dom_indices)
                    temp_row_player_df = temp_row_player_df.drop([dom_action for dom_action in dom_indices if dom_indices[dom_action]], axis = player_axis[get_other[solution.get('player_type')]])
                    temp_col_player_df = temp_col_player_df.drop([dom_action for dom_action in dom_indices if dom_indices[dom_action]], axis = player_axis[get_other[solution.get('player_type')]])
                    iterated_dfs.append(display_game_matrix(temp_row_player_df, temp_col_player_df))

        df_combined = display_game_matrix(row_player_df, col_player_df)
        
        options = [list(df_combined.index) if (player_axis[solution.get('player_type')] + i) % 2 == 0 else list(df_combined.columns) for i in range(len(iterated_actions))]
        # dominated_indices = [
        #     ALPHABET[get_option_letter(question_num = i, option_num = list(df_combined.index).index(dom_action), num_options = game_size[0])] 
        #     if (player_axis[solution.get('player_type')] + i) % 2 == 0 else 
        #     ALPHABET[get_option_letter(question_num = i, option_num = list(df_combined.columns).index(dom_action), num_options = game_size[1])]
        #     for i, dominated_actions in enumerate(iterated_actions) 
        #     for dom_action in dominated_actions if dominated_actions[dom_action]
        # ]
        return iterated_dfs, options#, dominated_indices


@function_registry.register('best_response')
def generate_best_response_values(metadata):
    """
    Best response in a 2x2 (difficulty 7) or 3x3 (difficulty 8) bimatrix game (recovered 2026-10; the old version
    was 2x2 only and offered both players' action labels as options).

    type = the side 'you' play; domain self: your best response to a stated action of your opponent, other: your
    opponent's best response to a stated action of yours. Payoffs are integers 0-19 or 2-decimal numbers in
    [0, 20), as generate_game_matrix; games where the responder has tied best responses are redrawn. Options: the
    responder's actions.
    """
    from .normal_form_tools import ROWS, COLS, payoffs, table, filler_utils as _fu
    rng = _fu.rng
    k = 3 if int(metadata['difficulty_level']) == 8 else 2
    you = metadata['type']
    responder = you if metadata['domain'] == 'self' else ('column' if you == 'row' else 'row')
    integer = bool(rng.random() < 0.5)
    while True:
        A, B = payoffs(k, k, 20, integer), payoffs(k, k, 20, integer)
        given = int(rng.integers(k))
        v = A[:, given] if responder == 'row' else B[given, :]
        if (v == v.max()).sum() == 1:
            break
    best = int(v.argmax())
    action = COLS[given] if responder == 'row' else ROWS[given]
    labels = ROWS[:k] if responder == 'row' else COLS[:k]
    return {
        'payoff_matrix': table(A, B, ROWS, COLS),
        'action': action,
        'options_list': [labels],
        'correct_answer': [[int(i == best) for i in range(k)]],
    }

################################################################
##                                                            ##
##                  Dominant Strategy Generator               ##
##                                                            ##
################################################################



def create_dominant_game(action_names, dominant_action):
    n = len(action_names['row'])
    m = len(action_names['column'])
    
    # Create empty payoff matrices
    row_player_df = np.zeros((n, m))
    col_player_df = np.zeros((n, m))
    
    # Set the dominant action payoff
    dominant_index = action_names['row'].index(dominant_action)
    for i in range(m):
        row_player_df[dominant_index][i] = 1
    
    # Set the other actions payoff
    for i in range(n):
        if i != dominant_index:
            for j in range(m):
                row_player_df[i][j] = 0
    
    # Create dataframes for display
    row_player_df = pd.DataFrame(row_player_df, index=[action_names['row'][i] for i in range(n)], columns=[action_names['column'][i] for i in range(m)])
    col_player_df = pd.DataFrame(col_player_df, index=[action_names['row'][i] for i in range(n)], columns=[action_names['column'][i] for i in range(m)])
    
    return row_player_df, col_player_df

# The word-problem fillers of dominant_strategies and pure_nash (generate_dominant_strategies_values,
# generate_nash_values) were replaced in October 2026 by code-written stories whose payoff table is generated
# first: see normal_form_stories_tools.py.


################################################################
##                                                            ##
##               Values Generator for Subgame                 ##
##                                                            ##
################################################################


def generate_subgame_values():
    # Generate random payoffs for each outcome
    X = random.randint(1, 10)  # Company A invests, Company B enters
    Y = random.randint(1, 10)  # Payoff for B when both enter
    Z = random.randint(1, 10)  # Company A invests, Company B does not enter
    W = random.randint(1, 10)  # Payoff for B when A invests and B does not enter
    P = random.randint(1, 10)  # Company A does not invest

    # Adjust payoffs to ensure different equilibria
    # For Equilibrium 1: A invests, B does not enter
    if Z <= P:
        Z = P + random.randint(1, 3)  # Ensure A prefers investing to not investing

    # For Equilibrium 2: A invests, B enters
    if X <= P or Y <= W:
        X = P + random.randint(1, 3)  # Ensure A prefers investing and B prefers entering
        Y = W + random.randint(1, 3)

    # For Equilibrium 3: A does not invest
    if P <= Z or P <= X:
        P = max(Z, X) + random.randint(1, 3)  # Ensure A prefers not investing

    return X, Y, Z, W, P


################################################################
##                                                            ##
##             Values Generator for Bayes-Nash                ##
##                                                            ##
################################################################


# ------------------------------------------------------------------ bayes_nash, Bayesian game with a typed opponent
BAYES_ROWS, BAYES_COLS = ['Action R', 'Action S'], ['Action K', 'Action L']


def bayes_typed_equilibria(U, V, p):
    """
    Pure Bayes-Nash equilibria of a 2x2 game where the column player (the opponent) has a private type t in {0, 1}
    with prior P(t = 0) = p and the row player ('you') has none. U[t], V[t] are the row / column payoffs (2x2) when
    the opponent is of type t. Returns [(x, (s0, s1))]: your action x and the opponent's type-contingent actions,
    by brute force over your 2 actions and the opponent's 4 strategies. Best responses must be strict.
    """
    out = []
    for x in range(2):
        for s in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mine = [p * U[0][a, s[0]] + (1 - p) * U[1][a, s[1]] for a in range(2)]
            if not mine[x] > mine[1 - x]:
                continue
            if all(V[t][x, s[t]] > V[t][x, 1 - s[t]] for t in range(2)):
                out.append((x, s))
    return out


def bayes_unique(U, V, p, tol=1e-9) -> bool:
    """
    True if the typed game (see bayes_typed_equilibria) has no Bayes-Nash equilibrium other than strict pure ones:
    nobody is indifferent at a pure profile (so every pure equilibrium is strict and there is no weak one), and
    there is no mixed equilibrium. Mixed equilibria: your mix q = P(R) makes an opponent type t indifferent only at
    its switching point q_t; there you are indifferent for some mix of type t exactly when your payoff difference
    R - S changes sign as type t moves from L to K (the other type's action is fixed by q_t). Both types switching
    at the same q_t, or you being indifferent against a pure strategy of the opponent, also count as not unique.
    """
    w = (p, 1 - p)

    def D(sig):   # your payoff R minus S when type t plays K with probability sig[t]
        return sum(w[t] * (sig[t] * (U[t][0, 0] - U[t][1, 0]) + (1 - sig[t]) * (U[t][0, 1] - U[t][1, 1]))
                   for t in range(2))

    def d(t, q):  # type t's payoff K minus L when you play R with probability q
        return q * (V[t][0, 0] - V[t][0, 1]) + (1 - q) * (V[t][1, 0] - V[t][1, 1])

    if any(abs(D((a, b))) < tol for a in (0, 1) for b in (0, 1)):
        return False
    if any(abs(d(t, q)) < tol for t in range(2) for q in (0.0, 1.0)):
        return False
    for t in range(2):
        a, b = d(t, 0.0), d(t, 1.0)
        if (a > 0) == (b > 0):
            continue                     # type t is never indifferent for q in [0, 1]
        q = a / (a - b)
        o = 1 - t
        if abs(d(o, q)) < tol:
            return False
        so = 1.0 if d(o, q) > 0 else 0.0
        lo = D((0.0, so) if t == 0 else (so, 0.0))
        hi = D((1.0, so) if t == 0 else (so, 1.0))
        if lo * hi < 0:
            return False                 # a mixed equilibrium at q
    return True


def _single_type_ne(U, V):
    """unique strict pure Nash equilibrium (row, col) of a complete-information 2x2 game, else None."""
    ne = [(i, j) for i in range(2) for j in range(2)
          if U[i, j] > U[1 - i, j] and V[i, j] > V[i, 1 - j]]
    return ne[0] if len(ne) == 1 else None


def _bayes_typed(metadata, rng):
    """
    type 'typed_own': your Bayes-Nash action; 'typed_opponent': the opponent's action for a stated type (drawn
    here, filled into {asked_type}). Payoffs are integers 0-9, the prior P(type 1) an integer percentage 10-90
    (not 50). Games are redrawn until there is exactly one pure BNE and no other equilibrium: no weak (tied best
    response) and no mixed one (bayes_unique; review, October 2026: a tie gave a second, weak equilibrium). In two thirds of the draws the game must also
    be one where the prior matters: the asked action differs from the answer of each single-type game (a single-
    type game without a unique strict pure equilibrium gives no answer, so it counts as different).
    """
    own = metadata['type'] == 'typed_own'
    want_matters = bool(rng.random() < 2 / 3)
    while True:
        U = [rng.integers(0, 10, size=(2, 2)).astype(float) for _ in range(2)]
        V = [rng.integers(0, 10, size=(2, 2)).astype(float) for _ in range(2)]
        pct = int(rng.integers(10, 91))
        if pct == 50:
            continue
        eq = bayes_typed_equilibria(U, V, pct / 100)
        if len(eq) != 1 or not bayes_unique(U, V, pct / 100):
            continue
        x, s = eq[0]
        t = int(rng.integers(2))
        singles = [_single_type_ne(U[k], V[k]) for k in range(2)]
        if own:
            matters = all(ne is None or ne[0] != x for ne in singles)
        else:
            matters = singles[t] is None or singles[t][1] != s[t]
        if matters or not want_matters:
            break

    def tab(k):
        cells = [[f'({int(U[k][i, j])}, {int(V[k][i, j])})' for j in range(2)] for i in range(2)]
        return pd.DataFrame(cells, index=BAYES_ROWS, columns=BAYES_COLS).to_string()
    if own:
        options, key = BAYES_ROWS, BAYES_ROWS[x]
    else:
        options, key = BAYES_COLS, BAYES_COLS[s[t]]
    return {
        'p1': pct, 'p2': 100 - pct, 'asked_type': f'Type {t + 1}',
        'type_tables': f'Type 1:\n{tab(0)}\nType 2:\n{tab(1)}',
        'options_list': [list(options)],
        'correct_answer': [[int(o == key) for o in options]],
    }


@function_registry.register('bayes_nash')
def generate_bayes_nash(metadata):
    """
    Types 'typed_own' / 'typed_opponent' (default since 2026-10-02): a genuine Bayesian game, see _bayes_typed.

    Types '<k>_states' (the recovered ICML design, kept but no longer in the template set; re-add with
    build_normal_form_templates.py --with-ex-ante): ex-ante Bayes-Nash equilibrium with 2-4 states
    (difficulty 10 + k) (recovered 2026-10 from the
    old generate_bayes_nash, which built the whole question itself and did not follow the sampler contract).

    As the original: a 2x2 base game (payoffs 2-decimal numbers in [0, 20)) with exactly one pure Nash equilibrium;
    state probabilities with 2 decimals summing to 1; each state's table is the base game times the state's
    probability, rounded to 2 decimals. Since no player observes the state, the game to solve is the expected
    game, i.e. (up to rounding) the base game. The draw is redone unless the expected game computed from the
    numbers shown has exactly one pure equilibrium, the base game's. Options: the four action pairs.
    """
    from .normal_form_tools import pure_nash, payoffs, filler_utils as _fu
    rng = _fu.rng
    if str(metadata.get('type', '')).startswith('typed_'):
        return _bayes_typed(metadata, rng)
    n = int(re.search(r'(\d+)_states', metadata['type']).group(1))
    rows, cols = ['Action R', 'Action S'], ['Action K', 'Action L']
    while True:
        A, B = payoffs(2, 2, 20, False), payoffs(2, 2, 20, False)
        ne = pure_nash(A, B)
        if len(ne) != 1:
            continue
        if n == 2:
            probs = [round(float(rng.uniform(0.1, 0.9)), 2)]
        else:
            probs = [round(float(rng.uniform(0.1, 0.9)), 2) for _ in range(n - 1)]
            if sum(probs) >= 0.99:
                continue
        probs.append(round(1 - sum(probs), 2))
        if abs(sum(probs) - 1) > 1e-9 or min(probs) <= 0:
            continue
        states = [(np.round(A * p, 2), np.round(B * p, 2)) for p in probs]
        EA = sum(p * a for p, (a, _) in zip(probs, states))
        EB = sum(p * b for p, (_, b) in zip(probs, states))
        if pure_nash(EA, EB) == ne:
            break
    blocks = []
    for i, (a, b) in enumerate(states):
        df = pd.DataFrame([[f'({a[r, c]}, {b[r, c]})' for c in range(2)] for r in range(2)], index=rows, columns=cols)
        blocks.append(f'State {i + 1}:\n' + df.to_string())
    options = [f"('{r}', '{c}')" for r in rows for c in cols]
    i, j = ne[0]
    return {
        'num_states': n,
        'states': ', '.join(f'State {i + 1}' for i in range(n)),
        'state_probs': ' and '.join(f'the probability of state {i + 1} is {p}' for i, p in enumerate(probs)).capitalize(),
        'state_tables': '\n'.join(blocks),
        'options_list': [options],
        'correct_answer': [[int(o == f"('{rows[i]}', '{cols[j]}')") for o in options]],
    }


################################################################
##                                                            ##
##        Repeated prisoner's dilemmas (shared helpers)       ##
##                                                            ##
################################################################
#
# trigger, enforceability and feasibility all describe a symmetric 2x2 stage game with a cooperative
# action C (collude, honor, keep prices high, ...) and a competitive action D. Own payoffs:
#   R = u(C, C), S = u(C, D), T = u(D, C), P = u(D, D).
# The templates' wording presumes a social dilemma, so every game is a genuine prisoner's dilemma:
# T > R > P > S and 2R > T + S. Then (D, D) is the unique stage-game Nash equilibrium and each
# player's pure minmax payoff is P.
#
# Placeholder -> role bindings. They follow the instructions the templates were generated from
# (template_configs/multi_agent.ipynb): trigger uses role names; enforceability/feasibility use
# both_x = both compete (P), both_y = both collude (R), x_payoff = the competitor (T),
# y_payoff = the cooperator (S). Templates whose wording binds them differently (or states a
# payoff as a cost/loss) are dropped at load time, see select_multi_agent_templates().

def sample_pd(low, high, integer=False, rel_gap=0.02):
    """Random PD payoffs {'R','S','T','P'} in [low, high], adjacent payoffs at least rel_gap*(high-low) apart."""
    gap = rel_gap * (high - low)
    while True:
        if integer:
            S, P, R, T = sorted(random.sample(range(low, high + 1), 4))
        else:
            S, P, R, T = sorted(round(random.uniform(low, high), 2) for _ in range(4))
        if min(P - S, R - P, T - R) >= gap and 2 * R - (T + S) >= gap:
            return {'R': R, 'S': S, 'T': T, 'P': P}


def grim_trigger_threshold(g):
    """Minimum delta for grim trigger (C until someone plays D, then D forever) to be an SPNE of the repeated PD:
    R/(1-d) >= T + d P/(1-d)  <=>  d >= (T - R) / (T - P)."""
    return (g['T'] - g['R']) / (g['T'] - g['P'])


def alternation_threshold(g):
    """Minimum delta for the take-turns path (C,D),(D,C),(C,D),... with grim-trigger reversion to (D,D) to be an
    SPNE. Per-period incentive constraints (normalised discounted values):
      period in which I cooperate (get S):  (S + d T)/(1 + d) >= P            (deviating gives P now and forever)
      period in which I defect (get T):     (T + d S)/(1 + d) >= (1-d) R + d P
    The second follows from the first when T > R, so the path is an SPNE iff d >= (P - S) / (T - P)."""
    return (g['P'] - g['S']) / (g['T'] - g['P'])


def money(v):
    return f"${v:.2f}"


def pd_money_placeholders(g):
    return {
        'both_x': money(g['P']),
        'both_y': money(g['R']),
        'x_payoff': money(g['T']),
        'y_payoff': money(g['S']),
    }


################################################################
##                                                            ##
##             Values Generator for Enforceability            ##
##                                                            ##
################################################################

# Option texts: each outcome is also identified by its per-period payoffs, so the (template-specific)
# action words are not needed to read it.
ENFORCEABILITY_OPTIONS = {
    'CC': "Both players cooperate every period, each earning {R} per period.",
    'DD': "Both players defect every period, each earning {P} per period.",
    'asym': "One player always defects while the other always cooperates, earning {T} and {S} per period respectively.",
    'alt': "The players take turns: each period one defects while the other cooperates, and they swap roles every period.",
    'none': "None of the other listed outcomes can be sustained.",
}


def enforceable_outcomes(g, delta):
    """Which outcomes are subgame-perfect equilibrium paths of the delta-discounted repeated PD with grim-trigger
    reversion to the stage-game Nash equilibrium (D, D) (Narun, 2026-09-27)."""
    return {
        'CC': delta >= grim_trigger_threshold(g),
        'DD': True,          # repeating the stage-game NE is always subgame perfect
        'asym': False,       # the cooperator earns S < P, their minmax: deviating to D forever pays P
        'alt': delta >= alternation_threshold(g),
    }


@function_registry.register('enforceability')
def generate_enforceability_values(metadata, margin=0.02):
    """
    Exactly one listed outcome is enforceable at the stated delta. (D, D) is always enforceable in a PD, so it is
    listed only when it is the answer; otherwise the fourth option is "none of the other listed outcomes".
      answer DD:   options CC, DD, asym, alt;   delta below both thresholds
      answer CC:   options CC, asym, alt, none; grim-trigger threshold <= delta < alternation threshold
      answer alt:  options CC, asym, alt, none; alternation threshold <= delta < grim-trigger threshold
      answer none: options CC, asym, alt, none; delta below both thresholds
    delta is kept at least `margin` away from both thresholds.
    """
    target = random.choice(['DD', 'CC', 'alt', 'none'])
    while True:
        g = sample_pd(10000, 100000)
        delta = round(random.uniform(0.1, 0.95), 2)
        d_cc, d_alt = grim_trigger_threshold(g), alternation_threshold(g)
        if min(abs(delta - d_cc), abs(delta - d_alt)) < margin:
            continue
        ok = enforceable_outcomes(g, delta)
        if target in ('DD', 'none'):
            good = not ok['CC'] and not ok['alt']
        elif target == 'CC':
            good = ok['CC'] and not ok['alt']
        else:
            good = ok['alt'] and not ok['CC']
        if good:
            break
    kinds = ['CC', 'DD', 'asym', 'alt'] if target == 'DD' else ['CC', 'asym', 'alt', 'none']
    shown = {k: money(v) for k, v in g.items()}
    options = [ENFORCEABILITY_OPTIONS[k].format(**shown) for k in kinds]
    return {
        **pd_money_placeholders(g),
        'discount_factor': f"{delta:.2f}",
        'options_list': [options],
        'correct_answer': [[k == target for k in kinds]],
    }


# feasibility: filler_tools/feasibility_tools.py (code templates, October 2026).


################################################################
##                                                            ##
##                Payoff Generator for Trigger                ##
##                                                            ##
################################################################

def trigger_distractors(g, correct, n=3):
    """Common wrong formulas for the grim-trigger threshold, then small perturbations; all in (0, 1), 2 decimals,
    at least 0.02 from the answer and from each other."""
    R, S, T, P = g['R'], g['S'], g['T'], g['P']
    cands = [(T - R) / (T - S), (R - P) / (T - P), (T - R) / (R - P), (T - R) / (R - S), (P - S) / (T - P),
             1 - correct, (T - R) / T]
    random.shuffle(cands)
    out = []

    def ok(v):
        return 0.01 <= v <= 0.99 and all(abs(v - w) >= 0.02 for w in [correct] + out)
    for v in cands:
        v = round(v, 2)
        if len(out) < n and ok(v):
            out.append(v)
    while len(out) < n:
        v = round(correct + random.choice([-1, 1]) * random.uniform(0.03, 0.15), 2)
        if ok(v):
            out.append(v)
    return out


def payoff_matrix_text(g, labels):
    """Complete 2x2 bimatrix (row payoff, column payoff) on its own lines; labels[0] is the cooperative action."""
    c, d = labels
    cells = [[f"({g['R']}, {g['R']})", f"({g['S']}, {g['T']})"],
             [f"({g['T']}, {g['S']})", f"({g['P']}, {g['P']})"]]
    return '\n' + pd.DataFrame(cells, index=[c, d], columns=[c, d]).to_string() + '\n'


@function_registry.register('trigger')
def gen_trigger(metadata):
    """Minimum discount factor for grim trigger in a repeated PD: (T - R)/(T - P), 2 decimals."""
    while True:
        g = sample_pd(50, 1000, integer=True)
        discount = round(grim_trigger_threshold(g), 2)
        if 0.05 <= discount <= 0.95:
            break
    options = [discount] + trigger_distractors(g, discount)
    options_list = [f"{v:.2f}" for v in options]
    if 'explicit' in str(metadata.get('tags', '')).split(';'):
        labels = ['Cooperate', 'Defect']
    else:
        letters = random.sample(ALPHABET, 2)
        labels = ['Action ' + letters[0], 'Action ' + letters[1]]
    return {
        'payoff_matrix': payoff_matrix_text(g, labels),
        'action_1': labels[0],
        'action_2': labels[1],
        'both_cooperate': g['R'],
        'cooperate': g['S'],
        'defect': g['T'],
        'both_defect': g['P'],
        'options_list': [options_list],
        'correct_answer': [[i == 0 for i in range(len(options_list))]],
    }


################################################################
##                                                            ##
##            Values Generator for Backward Induction         ##
##                                                            ##
################################################################

@function_registry.register('backward_induction')
def gen_backward_induction_values(metadata):
    """
    Three-round centipede-style game: round 1 player 1 ends or continues, round 2 player 2, round 3 player 1
    ends ({p?_round3}) or lets the third party set the terms ({p?_round3_2}). The template's options are fixed
    and its correct option is the tag's 'continue'/'end' choice at the tag's round, so the game is redrawn until
    backward induction gives that choice, with no indifference at any node.
    """
    if metadata['domain'] == 'medical':
        payoff_type, payoff_range = int, (-10, 10)
    else:
        payoff_type, payoff_range = float, (10000, 100000)
    tags = str(metadata['tags']).split(';')
    round_num = 1 if 'round1' in tags else 2
    want = 'right' if 'continue' in tags else 'left'
    while True:
        game_tree = GameTree(3, 2, payoff_type, payoff_range)
        if game_tree.compute_backwards_induction(round_num) == want:
            break
    last = game_tree.get_decision_node(2)
    values = {
        'p1_round1': game_tree.get_decision_node(0).left.get_payoffs(1),
        'p2_round1': game_tree.get_decision_node(0).left.get_payoffs(2),
        'p1_round2': game_tree.get_decision_node(1).left.get_payoffs(1),
        'p2_round2': game_tree.get_decision_node(1).left.get_payoffs(2),
        'p1_round3': last.left.get_payoffs(1),
        'p2_round3': last.left.get_payoffs(2),
        'p1_round3_2': last.right.get_payoffs(1),
        'p2_round3_2': last.right.get_payoffs(2),
    }
    return {k: v.item() for k, v in values.items()}   # numpy scalars -> int / float


################################################################
##                                                            ##
##          Template selection for the four elements          ##
##                                                            ##
################################################################

MULTI_AGENT_TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), 'multi_agent_templates.json')
_selected_templates = None


def template_key(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def _load_selected():
    global _selected_templates
    if _selected_templates is None:
        if os.path.exists(MULTI_AGENT_TEMPLATES_PATH):
            with open(MULTI_AGENT_TEMPLATES_PATH) as f:
                _selected_templates = {e: set(keys) for e, keys in json.load(f)['elements'].items()}
        else:
            _selected_templates = {}
    return _selected_templates


PD_ROLE_FIXES_PATH = os.path.join(os.path.dirname(__file__), 'pd_role_fixes.json')
_pd_role_fixes = None
_PD_SWAP = {'both_x': 'both_y', 'both_y': 'both_x', 'x_payoff': 'y_payoff', 'y_payoff': 'x_payoff',
            'both_cooperate': 'both_defect', 'both_defect': 'both_cooperate', 'cooperate': 'defect', 'defect': 'cooperate'}
_PD_PH = re.compile(r'\{(' + '|'.join(_PD_SWAP) + r')\}')


def _load_pd_role_fixes():
    global _pd_role_fixes
    if _pd_role_fixes is None:
        with open(PD_ROLE_FIXES_PATH) as f:
            _pd_role_fixes = json.load(f)['elements']
    return _pd_role_fixes


def swap_pd_roles(text: str) -> str:
    """Exchange the cooperate and defect placeholders (both_x <-> both_y, x_payoff <-> y_payoff, ...)."""
    return _PD_PH.sub(lambda m: '{' + _PD_SWAP[m.group(1)] + '}', text)


_MATRIX = re.compile(r'(\b(?:in|by)\s*)?:?\s*\{payoff_matrix\}\s*([.,]?)')


def matrix_last(text: str) -> str:
    """The payoff table on its own lines at the end: "Given the payoff matrix {payoff_matrix}, what is ..." printed
    the table inside the sentence ("matrix \\n<table>\\n, what is"). The sentence keeps "below" / "the matrix below"."""
    if '{payoff_matrix}' not in text:
        return text

    def repl(m):
        lead = (m.group(1) or '') + ('the matrix below' if m.group(1) else '')
        if m.group(2) == ',':
            return lead + (' below,' if not m.group(1) else ',')
        if not m.group(2) and ':' in m.group(0) and m.end() == len(text.rstrip()):
            return lead + ':'      # "is as follows: {payoff_matrix}" at the end
        return lead + m.group(2)
    return _MATRIX.sub(repl, text, count=1).rstrip() + '\n\n{payoff_matrix}'


def apply_pd_role_fixes(element, q, opts):
    """Templates whose story binds the payoff placeholders to the other action (pd_role_fixes.json, built by
    scripts/build_pd_role_fixes.py from the raw template texts): 'swap' ones get the placeholders exchanged, 'drop'
    ones are removed. Two spaces after a punctuation mark become one."""
    fixes = _load_pd_role_fixes().get(element)
    if not fixes or q is None or len(q) == 0:
        return q, opts
    keys = q['question_text'].map(template_key)
    keep = ~keys.isin(set(fixes['drop']))
    swap = keys[keep].isin(set(fixes['swap']))
    q = q[keep].copy()
    q.loc[swap, 'question_text'] = q.loc[swap, 'question_text'].map(swap_pd_roles)
    q['question_text'] = q['question_text'].str.replace(r'(?<=[.!?,;:])  +(?=\S)', ' ', regex=True)
    if element == 'trigger':
        q['question_text'] = q['question_text'].map(matrix_last).map(trigger_rounding)
    q = q.reset_index(drop=True)
    if opts is not None and len(opts):
        opts = opts[opts['question_id'].isin(set(q['question_id']))].reset_index(drop=True)
    return q, opts


# Templates dropped after the review (October 2026), by template_key of the selected text, with the reason.
TEMPLATE_DROPS = {
    'trigger': {
        'b0676a8155907495': 'the player who secures gets the defector payoff ({defect} / {cooperate} respectively)',
        '889b95d89b92efe1': 'the "secure-gambler" gets the defector payoff; securing would dominate',
        '23ad66e1ba2bfc8b': '"one secures and the other gambles ({defect}, {cooperate})": securer gets the defector payoff',
        'acad02c650198b1d': '"the former" (who secures) gets the defector payoff',
        'e6df2f622c866eba': 'the risk-taker "either loses everything or comes out with" the payoff: not a stage payoff',
        '569b50b6f3fdc069': 'the risk-taker "may end up with" the payoff: not a stage payoff',
        '568703b5e93d3402': 'the safe player gets the defector payoff ({defect} and {cooperate}, respectively)',
        'f129b5faa392abd1': '"release proprietary code" vs "share proprietary code": which action is cooperation is unclear',
        'b97421fc97575339': '"the grim trigger strategy of always collaborating" names unconditional cooperation',
        '6f46a76a6c960764': '"the strategy of always settling" names unconditional cooperation',
    },
    'backward_induction': {
        '7390fe7076dd9e8d': 'keyed "continue with her current plan" is the ending action (its payoffs are round 1)',
        '120577ac12d91f79': 'keyed "continue with her own protocol" is the ending action; round 2 moved by the wrong doctor',
        '41d9ff3d9c5d98ce': 'the third-round mover is unclear (Dr. Lee "switches back" in Dr. Smith\'s round)',
        '56b071d82942ac2e': 'round-1 payoffs attached to both actions ("adopt ... or defer ..., resulting in")',
        'a6c18ae45e6b06df': 'round 2 both ends the game and leads to round 3 ("agree to continue")',
        '1bc07a746c794b14': '"stick with" both ends the game in round 2 and leads to round 3',
        '60dc365186815468': 'keyed "continue with her treatment" is the ending action; third round moved by the wrong doctor',
        '7780ea9d8c68abd9': 'round-1 payoffs attached to the second mover\'s choice',
        '428f492c46009c05': 'keyed "continue with her own treatment plan" is the ending action',
        '9251adb66def3f7d': 'payoffs printed as equity percentages ("23442.69%")',
        '6ccf1d5679ff4830': 'payoffs printed as equity percentages',
    },
}
_BI_IMPROVE_CLAIM = re.compile(r'\s*[^.!?]*\bimproves?\b[^.!?]*\bas the rounds\b[^.!?]*[.!?]')
_BI_IMPROVED = re.compile(r'\bimproved (terms|offer|deal|conditions)\b')


def bi_wording(text: str) -> str:
    """backward_induction stories must not assert that later terms are better (the payoffs are drawn at random):
    "The conditions of the deal improve for both investors as the rounds advance ..." is removed and "the improved
    terms" becomes "the new terms"."""
    return _BI_IMPROVED.sub(r'new \1', _BI_IMPROVE_CLAIM.sub('', text))


_TRAILING_MATRIX = '\n\n{payoff_matrix}'


def trigger_rounding(text: str) -> str:
    """State the rounding of the threshold (the key is (T - R)/(T - P) rounded to 2 decimals): after the question's
    last '?' (the payoff table, placed last by matrix_last, stays last)."""
    body, tail = (text[:-len(_TRAILING_MATRIX)], _TRAILING_MATRIX) if text.endswith(_TRAILING_MATRIX) else (text, '')
    note = 'Round your answer to two decimal places.'
    i = body.rfind('?')
    body = body[:i + 1] + ' ' + note + body[i + 1:] if i >= 0 else body.rstrip() + ' ' + note
    return body + tail


def _flip_bi_tag(tags: str) -> str:
    parts = str(tags).split(';')
    return ';'.join({'continue': 'end', 'end': 'continue'}.get(p, p) for p in parts)


def select_multi_agent_templates(element, q, opts):
    """
    Load-time template selection for trigger / enforceability / feasibility / backward_induction (called from
    src/autosteer/sampler/template_fixes.py). Keeps only the templates listed in multi_agent_templates.json: those whose
    wording binds the payoff placeholders to the roles the filler assumes, with no payoff stated as a cost, all
    payoffs present, and (backward_induction) the asked round and the options matching the tag. The list is
    built by audit/steer/select_multi_agent_templates.py, which reads filled templates with the independent audit
    reader (audit/steer/solvers/repeated_games.py).

    enforceability: the answer is computed from the game, so the answer-describing tags (both_high, one_honor, ...)
    are dropped. backward_induction: every template is also cloned with the opposite tag (continue <-> end) and
    its option flags flipped, so both answers occur in every domain (medical only had 'continue' templates).
    """
    keep = _load_selected().get(element)
    if keep is None or q is None or len(q) == 0:
        return q, opts
    keep = keep - set(TEMPLATE_DROPS.get(element, {}))
    q = q[q['question_text'].map(template_key).isin(keep)].reset_index(drop=True)
    ids = set(q['question_id'])
    if opts is not None and len(opts):
        opts = opts[opts['question_id'].isin(ids)].reset_index(drop=True)
    q, opts = apply_pd_role_fixes(element, q, opts)
    if element == 'enforceability':
        q = q.copy()
        q['tags'] = q['tags'].map(lambda t: str(t).split(';')[0])
    if element == 'backward_induction' and len(q):
        q = q.copy()
        q['question_text'] = q['question_text'].map(bi_wording)
        offset = int(q['question_id'].max()) + 1
        flipped = q.copy()
        flipped['question_id'] = flipped['question_id'] + offset
        flipped['tags'] = flipped['tags'].map(_flip_bi_tag)
        q = pd.concat([q, flipped], ignore_index=True)
        if opts is not None and len(opts):
            fo = opts.copy()
            fo['question_id'] = fo['question_id'] + offset
            fo['correct_answer'] = ~fo['correct_answer'].astype(bool)
            opts = pd.concat([opts, fo], ignore_index=True)
    return q, opts
