"""Independent answer oracle for sampled trigger / enforceability / feasibility / backward_induction questions.

Wraps the audit solvers in solvers/repeated_games.py (which read the game from the question text and options),
with two adaptations for the regenerated questions:
  - trigger: the explicit payoff matrices label their rows 'Cooperate' / 'Defect'; the audit matrix reader expects
    'Action X' labels, so the labels are prefixed before reading.
  - enforceability: the audit solver leaves rows whose take-turns option is enforceable open (semantics were
    undecided) and has no 'none of the other listed outcomes' option. Narun's decision (take turns = the players
    alternate the aggressor role) is what solvers.repeated_games.enforceable() already evaluates, so this oracle
    applies it to every option, with 'none' correct iff no other listed option is enforceable.

oracle(element, question, options, tags, domain) -> (index of the correct option or None, status).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


def _load_repeated_games():
    # by file path: tests also put audit/steer_me/solvers on sys.path as the package name `solvers`
    name = "steer_audit_repeated_games"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "solvers" / "repeated_games.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


RG = _load_repeated_games()

_NONE = re.compile(r"^\s*none of\b", re.I)


def _one(element, question, options, tags, domain):
    df = pd.DataFrame({"question_text": [question], "options": [list(options)],
                       "tags": [list(tags) if tags is not None else []], "domain": [domain]})
    r = RG.SOLVERS[element](df).iloc[0]
    return (None if pd.isna(r["pred"]) else int(r["pred"])), r["status"]


def _enforceability(question, options):
    g, why = RG.parse_game(question)
    if g is None:
        return None, f"unparsed:{why}"
    delta = RG.discount(question)
    if delta is None:
        return None, "unparsed:discount"
    e = RG.enforceable(g, delta)
    if e is None:
        return None, "ill_posed:no_pure_stage_ne"
    kinds = ["none" if _NONE.match(o) else RG.enf_option_kind(o) for o in options]
    if None in kinds or len(set(kinds)) != len(kinds):
        return None, "unparsed:options"
    listed = {k: e[k] for k in kinds if k != "none"}
    if "none" in kinds:
        e = dict(e, none=not any(listed.values()))
    hit = [i for i, k in enumerate(kinds) if e[k]]
    if len(hit) != 1:
        return None, f"ill_posed:{len(hit)}_enforceable"
    return hit[0], "ok"


def oracle(element, question, options, tags=None, domain=None):
    if isinstance(tags, str):
        tags = tags.split(";")
    if element == "enforceability":
        return _enforceability(question, options)
    if element == "trigger":
        question = re.sub(r"^(\s*)(Cooperate|Defect)(\s+\()", r"\1Action \2\3", question, flags=re.M)
    return _one(element, question, options, tags, domain)
