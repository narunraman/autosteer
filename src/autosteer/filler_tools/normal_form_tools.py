"""Fillers for the STEER normal-form matrix elements recovered from the v1 / generation-repo code (2026-10).

    interpret_games       read one cell of a 2x2 / 3x3 payoff table          (rb fill_generations/dom_game_gen.ipynb)
    dominated_strategies  pick the strictly dominant action                   (v1 utils.add_dom, dom_game_gen.ipynb)
    iterated_removal      two rounds of iterated strict dominance             (multi_agent_tools.generate_game)
    subgame_nash          finitely repeated stage game: repeat its equilibrium (generate_game 'nash')

Templates (data/templates/<element>/, Parquet) are the released sentences, recovered verbatim by
audit/steer/regen/build_normal_form_templates.py; the table, the actions named in the sentence and the options come
from here. Payoffs follow the original generators: v1 elements (interpret_games, dominated_strategies) draw integers
0-9 or 2-decimal numbers in [0, 10), plus 1-3 on the dominant action; generation-repo elements draw integers 0-19 or
2-decimal numbers in [0, 20). Integer and decimal tables are equally likely.

Answers are computed here and re-checked by brute force in the tests (tests/test_normal_form_games.py) and by the
independent audit solvers (audit/steer/solvers/matrix_games.py). Draws that would make a question ambiguous
(tied best responses, no or several dominant / dominated actions, stage games with more than one equilibrium) are
redrawn. All randomness comes from filler_utils.rng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ROWS_V1, COLS_V1 = ['Action R', 'Action S', 'Action T'], ['Action X', 'Action Y', 'Action Z']
ROWS, COLS = ['Action R', 'Action S', 'Action T'], ['Action K', 'Action L', 'Action M']
ROUND_WORDS = {'two': 2, 'three': 3, 'four': 4}


def _rng():
    return filler_utils.rng


def payoffs(n: int, m: int, hi: int, integer: bool) -> np.ndarray:
    if integer:
        return _rng().integers(0, hi, size=(n, m)).astype(float)
    return np.round(_rng().random((n, m)) * hi, 2)


def fmt(v: float) -> str:
    """Payoff as shown: integers without decimals, others with at most 2 decimals."""
    v = round(float(v), 2)
    return str(int(v)) if v == int(v) else repr(v)


def table(A: np.ndarray, B: np.ndarray | None, rows: list, cols: list) -> str:
    """pandas-rendered payoff table: '(a, b)' cells (row player first) or single payoffs."""
    n, m = A.shape
    if B is None:
        cells = [[fmt(A[i, j]) for j in range(m)] for i in range(n)]
    else:
        cells = [[f'({fmt(A[i, j])}, {fmt(B[i, j])})' for j in range(m)] for i in range(n)]
    return pd.DataFrame(cells, index=rows[:n], columns=cols[:m]).to_string()


def strictly_dominant(U: np.ndarray) -> list:
    """rows of U (the chooser's actions as rows) that strictly dominate every other row."""
    return [a for a in range(U.shape[0]) if all((U[a] > U[b]).all() for b in range(U.shape[0]) if b != a)]


def strictly_dominated(U: np.ndarray) -> list:
    return [a for a in range(U.shape[0]) if any((U[b] > U[a]).all() for b in range(U.shape[0]) if b != a)]


def dominated_any(U: np.ndarray, tol: float = 1e-9) -> list:
    """Actions (rows of U) that are dominated in any usual sense: weakly by another pure action (never worse,
    sometimes better) or strictly by a mixture of the other actions (linear program). Used to make sure the asked
    dominated action is the only candidate whichever notion of dominance the reader uses (review, Oct 2026)."""
    from scipy.optimize import linprog
    k, m = U.shape
    out = []
    for a in range(k):
        others = [b for b in range(k) if b != a]
        if any((U[b] >= U[a]).all() and (U[b] > U[a]).any() for b in others):
            out.append(a)
            continue
        # max eps s.t. sum_b x_b U[b, j] >= U[a, j] + eps for all j, x >= 0, sum x = 1
        c = np.zeros(len(others) + 1)
        c[-1] = -1.0
        A_ub = np.hstack([-U[others].T, np.ones((m, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=-U[a], A_eq=[[1.0] * len(others) + [0.0]], b_eq=[1.0],
                      bounds=[(0, None)] * len(others) + [(None, None)], method='highs')
        if res.status == 0 and -res.fun > tol:
            out.append(a)
    return out


def pure_nash(A: np.ndarray, B: np.ndarray) -> list:
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] >= A[:, j].max() and B[i, j] >= B[i, :].max()]


def iesds(A: np.ndarray, B: np.ndarray):
    """Iterated elimination of strictly dominated actions; returns surviving (rows, cols)."""
    r, c = list(range(A.shape[0])), list(range(A.shape[1]))
    changed = True
    while changed:
        changed = False
        for i in list(r):
            if any((A[k, c] > A[i, c]).all() for k in r if k != i):
                r.remove(i)
                changed = True
        for j in list(c):
            if any((B[r, k] > B[r, j]).all() for k in c if k != j):
                c.remove(j)
                changed = True
    return r, c


def _player(metadata) -> str:
    return metadata['type']  # 'row' / 'column': the side 'you' play


def _other(p: str) -> str:
    return 'column' if p == 'row' else 'row'


def _size(metadata, grade_of_3x3) -> int:
    return 3 if int(metadata['difficulty_level']) in grade_of_3x3 else 2


def _tags(metadata) -> list:
    return str(metadata.get('tags', '')).split(';')


# ------------------------------------------------------------------ interpret_games
@function_registry.register('interpret_games')
def fill_interpret_games(metadata: dict) -> dict:
    """type = your side, domain = whose payoff is asked (self / other); difficulty 0 = 2x2, 1 = 3x3.

    The template says whether a bimatrix or the asked player's own matrix is shown (single tables are always the
    asked player's). Options: the asked player's payoffs on a random transversal through the asked cell (the
    released layout), distinct values."""
    k = _size(metadata, {1})
    you = _player(metadata)
    owner = you if metadata['domain'] == 'self' else _other(you)
    single = 'find your' in str(metadata.get('question_text', ''))
    integer = bool(_rng().random() < 0.5)
    while True:
        A, B = payoffs(k, k, 10, integer), payoffs(k, k, 10, integer)
        U = A if owner == 'row' else B
        i, j = int(_rng().integers(k)), int(_rng().integers(k))
        perm = list(_rng().permutation(k))
        perm[perm.index(j)], perm[i] = perm[i], j   # transversal through (i, j)
        vals = [U[r, perm[r]] for r in range(k)]
        if len({fmt(v) for v in vals}) == k:
            break
    mine, theirs = (ROWS_V1[i], COLS_V1[j]) if you == 'row' else (COLS_V1[j], ROWS_V1[i])
    shown = table(U, None, ROWS_V1, COLS_V1) if single else table(A, B, ROWS_V1, COLS_V1)
    opts = [fmt(v) for v in vals]   # as the table shows them ("7", not "7.0")
    return {'payoff_matrix': shown, 'my_action': mine, 'their_action': theirs,
            'options_list': [opts], 'correct_answer': [[int(r == i) for r in range(k)]]}


# ------------------------------------------------------------------ dominated_strategies
@function_registry.register('dominated_strategies')
def fill_dominated_strategies(metadata: dict) -> dict:
    """type = who chooses (self / other) and which action is asked for: 'self' / 'other' ask for the strictly
    dominant action (the released questions), 'self_dominated' / 'other_dominated' for the strictly dominated
    action, i.e. the one never to play (added 2026-10-02); no other action may be weakly dominated by a pure action
    or strictly dominated by a mixture (review, October 2026: a weakly dominated second action). tags = your side (row / column); difficulty 0 = 2x2,
    1 = 3x3. Single tables show the chooser's own payoffs. Options: the chooser's actions.

    Dominant kind, as v1 add_dom: one action is set to the element-wise maximum of the others plus 1-3, so it
    strictly dominates every other action. Dominated kind: one action is set to another action's payoffs minus
    1-3 (element-wise), so it is strictly dominated; the draw is redone unless it is the only strictly dominated
    action."""
    k = _size(metadata, {1})
    you = _tags(metadata)[0]
    kind = 'dominated' if str(metadata['type']).endswith('_dominated') else 'dominant'
    chooser = you if str(metadata['type']).startswith('self') else _other(you)
    single = 'find your' in str(metadata.get('question_text', ''))
    integer = bool(_rng().random() < 0.5)
    while True:
        A, B = payoffs(k, k, 10, integer), payoffs(k, k, 10, integer)
        U = A if chooser == 'row' else B.T          # chooser's actions as rows
        d = int(_rng().integers(k))
        if kind == 'dominant':
            U[d] = np.round(U.max(axis=0) + int(_rng().integers(1, 4)), 2)
            ok = strictly_dominant(U) == [d]
        else:
            b = int(_rng().choice([x for x in range(k) if x != d]))
            U[d] = np.round(U[b] - _rng().integers(1, 4, size=k), 2)
            ok = strictly_dominated(U) == [d] and dominated_any(U) == [d] and (U[d] >= 0).all()
        if ok:
            break
    if chooser == 'row':
        A = U
    else:
        B = U.T
    shown = table(U if chooser == 'row' else U.T, None, ROWS_V1, COLS_V1) if single else table(A, B, ROWS_V1, COLS_V1)
    labels = ROWS_V1[:k] if chooser == 'row' else COLS_V1[:k]
    return {'payoff_matrix': shown, 'options_list': [labels], 'correct_answer': [[int(a == d) for a in range(k)]]}


# ------------------------------------------------------------------ iterated_removal
@function_registry.register('iterated_removal')
def fill_iterated_removal(metadata: dict) -> dict:
    """Two parts on a 3x3 bimatrix. domain self: you have the first dominated action, other: your opponent.
    The game is redrawn until the first player has exactly one strictly dominated action and, once it is
    removed, the second player has exactly one (as generate_game's 'iterated' branch); in both steps no other action
    is weakly dominated by a pure action or strictly dominated by a mixture (dominated_any), and the second player
    has no dominated action at all in the original game (the first removal is needed). Difficulty 5 shows the
    reduced table in part 2 ({reduced_matrix}); 6 and 7 only ask."""
    you = _player(metadata)
    first = you if metadata['domain'] == 'self' else _other(you)
    second = _other(first)
    integer = bool(_rng().random() < 0.5)
    while True:
        A, B = payoffs(3, 3, 20, integer), payoffs(3, 3, 20, integer)
        U1 = A if first == 'row' else B.T
        d1 = strictly_dominated(U1)
        if len(d1) != 1 or dominated_any(U1) != d1:
            continue
        keep = [x for x in range(3) if x != d1[0]]
        A2, B2 = (A[keep], B[keep]) if first == 'row' else (A[:, keep], B[:, keep])
        U2 = A2 if second == 'row' else B2.T
        d2 = strictly_dominated(U2)
        # the second player has no dominated action before the first removal, so part 2 needs it (re-review,
        # October 2026: the part-2 action was sometimes dominated in the original game already)
        if len(d2) == 1 and dominated_any(U2) == d2 and not dominated_any(A if second == 'row' else B.T):
            break
    rows2 = [ROWS[x] for x in keep] if first == 'row' else ROWS
    cols2 = COLS if first == 'row' else [COLS[x] for x in keep]
    lab1 = ROWS if first == 'row' else COLS
    lab2 = ROWS if second == 'row' else COLS
    return {'payoff_matrix': table(A, B, ROWS, COLS),
            'reduced_matrix': pd.DataFrame([[f'({fmt(A2[i, j])}, {fmt(B2[i, j])})' for j in range(A2.shape[1])]
                                            for i in range(A2.shape[0])], index=rows2, columns=cols2).to_string(),
            'options_list': [lab1, lab2],
            'correct_answer': [[int(a == d1[0]) for a in range(3)], [int(a == d2[0]) for a in range(3)]]}


# ------------------------------------------------------------------ subgame_nash
def _rounds_text(moves) -> str:
    """'Round 1: you play Action S, your opponent plays Action K; round 2: ...' from one clause per round."""
    return '; '.join(f'{"R" if t == 0 else "r"}ound {t + 1}: {m}' for t, m in enumerate(moves))


@function_registry.register('subgame_nash')
def fill_subgame_nash(metadata: dict) -> dict:
    """A 2x2 or 3x3 stage game repeated two to four rounds; domain self / other / both = your actions, your
    opponent's, or the action pairs. The rounds word and the size come from the template text and difficulty
    (8 + rounds, + 1 for 3x3). The stage game is redrawn until iterated strict dominance leaves a single cell, so
    it has a unique Nash equilibrium (pure, and no mixed one) and the finitely repeated game has a unique subgame-
    perfect equilibrium: that cell in every round. Options as released: every action (or action pair) repeated in
    every round, written per round in plain words ("Round 1: you play Action S, your opponent plays Action K;
    round 2: ..."; your action first)."""
    text = str(metadata.get('question_text', ''))
    n_rounds = next((v for w, v in ROUND_WORDS.items() if f'for {w} rounds' in text), 2)
    k = 3 if int(metadata['difficulty_level']) - 8 - n_rounds == 1 else 2
    you = _player(metadata)
    integer = bool(_rng().random() < 0.5)
    while True:
        A, B = payoffs(k, k, 20, integer), payoffs(k, k, 20, integer)
        r, c = iesds(A, B)
        if len(r) == 1 and len(c) == 1:
            break
    i, j = r[0], c[0]
    assert pure_nash(A, B) == [(i, j)]
    dom = metadata['domain']
    if dom == 'both':
        pairs = [(a, b) for a in range(k) for b in range(k)]
        def clause(a, b):
            mine, theirs = (ROWS[a], COLS[b]) if you == 'row' else (COLS[b], ROWS[a])
            return f'you play {mine}, your opponent plays {theirs}'
        opts = [_rounds_text([clause(a, b)] * n_rounds) for a, b in pairs]
        correct = [int((a, b) == (i, j)) for a, b in pairs]
    else:
        p = you if dom == 'self' else _other(you)
        labels = ROWS[:k] if p == 'row' else COLS[:k]
        who = 'you play' if dom == 'self' else 'your opponent plays'
        opts = [_rounds_text([f'{who} {lab}'] * n_rounds) for lab in labels]
        target = i if p == 'row' else j
        correct = [int(x == target) for x in range(k)]
    return {'payoff_matrix': table(A, B, ROWS, COLS), 'options_list': [opts], 'correct_answer': [correct]}
