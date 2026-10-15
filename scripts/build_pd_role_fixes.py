"""Build filler_tools/pd_role_fixes.json: per-template corrections of the payoff-placeholder roles of the
repeated prisoner's dilemma elements (feasibility, enforceability, trigger).

The fillers bind both_y / both_cooperate = R (both cooperate), both_x / both_defect = P (both defect),
x_payoff / defect = T (the defector against a cooperator), y_payoff / cooperate = S (the cooperator).
Each selected template is filled with sentinel values in that binding and read back with the independent
story reader of the tests (tests/solvers/steer/solvers/repeated_games.py:parse_game):

- the reader recovers the binding: kept as is;
- the story tells the binding fully swapped (both_x is the both-cooperate payoff, y_payoff the defector's,
  ...: 289 feasibility templates): 'swap', the placeholders are exchanged (both_x <-> both_y,
  x_payoff <-> y_payoff) at load time, so the story's cooperate action carries R;
- any other reading, a payoff stated as a loss or negative ("incurs a loss of {x_payoff}"; the filler's payoffs
  are positive), or no reading: 'drop'.

Run: AUTOSTEER_DATA_DIR=... .venv/bin/python scripts/build_pd_role_fixes.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'solvers' / 'steer'))
from solvers.repeated_games import parse_game  # noqa: E402

from autosteer.filler_tools import multi_agent_tools  # noqa: E402
from autosteer.filler_tools.multi_agent_tools import template_key  # noqa: E402
from autosteer.sampler.generator import ElementSampler  # noqa: E402

OUT = ROOT / 'src' / 'autosteer' / 'filler_tools' / 'pd_role_fixes.json'
BINDINGS = {
    'feasibility': {'both_y': 'R', 'both_x': 'P', 'x_payoff': 'T', 'y_payoff': 'S'},
    'enforceability': {'both_y': 'R', 'both_x': 'P', 'x_payoff': 'T', 'y_payoff': 'S'},
    'trigger': {'both_cooperate': 'R', 'both_defect': 'P', 'defect': 'T', 'cooperate': 'S'},
}
SENTINEL = {'R': 51111, 'P': 32222, 'T': 74444, 'S': 13333}
SMALL = {'R': 511, 'P': 322, 'T': 744, 'S': 133}
SWAPPED = {'R': 'P', 'P': 'R', 'T': 'S', 'S': 'T'}


def fill(text, binding, money):
    vals = SENTINEL if money else SMALL
    for name, role in binding.items():
        text = text.replace('{' + name + '}', f'${vals[role]:,}' if money else str(vals[role]))
    return text.replace('{discount_factor}', '0.55')


def classify(text, binding, money):
    """'keep', 'swap' or a drop reason."""
    if '{payoff_matrix}' in text:
        return 'keep'
    g, why = parse_game(fill(text, binding, money))
    if g is None:
        return 'unread:' + why
    if any(v < 0 for v in g.values()):
        return 'payoff stated as a loss'
    vals = SENTINEL if money else SMALL
    role_of = {v: r for r, v in vals.items()}
    got = {story: role_of.get(int(v)) for story, v in g.items()}
    if all(got[r] == r for r in 'RSTP'):
        return 'keep'
    if all(got[r] == SWAPPED[r] for r in 'RSTP'):
        return 'swap'
    return 'roles ' + ''.join(str(got[r]) for r in 'RSTP')


def main():
    multi_agent_tools._pd_role_fixes = {}   # read the templates without the corrections this script builds
    sampler = ElementSampler()
    fixes = {}
    for element, binding in BINDINGS.items():
        q, _ = sampler._pool_frames(sampler._load_template(element), 'questions')
        swap, drop = [], {}
        for text in q['question_text']:
            c = classify(text, binding, money=element != 'trigger')
            if c == 'swap':
                swap.append(template_key(text))
            elif c != 'keep':
                drop[template_key(text)] = c
        fixes[element] = {'swap': sorted(set(swap)), 'drop': dict(sorted(drop.items()))}
        print(element, len(q), 'swap', len(set(swap)), 'drop', len(drop))
    OUT.write_text(json.dumps({'generator': 'scripts/build_pd_role_fixes.py', 'elements': fixes}, indent=1) + '\n')


if __name__ == '__main__':
    main()
