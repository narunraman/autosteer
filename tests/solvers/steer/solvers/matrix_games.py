"""Solvers and repair hooks for the normal-form / matrix-game elements.

Elements: best_response, dominated_strategies, interpret_games, iterated_removal, pure_nash, subgame_nash,
dominant_strategies, bayes_nash.

Matrix elements print a pandas payoff table after the question: bimatrix cells "(a, b)" (a = row player's payoff,
b = column player's) or single-payoff tables ("your payoff matrix" / "your opponent's payoff matrix"; the table is
then the payoffs of the player named in the text).  Row labels are Action R/S/T(/U), column labels Action K/L/M
(or X/Y/Z), so every action label identifies its player.  "You are the row|column player" fixes the perspective;
"your opponent" is the other player.

Word-problem elements (dominant_strategies, the 6,597 generation-repo rows of pure_nash) are LLM-written templates
filled by template_configs/multi_agent_utils.py (generate_dominant_strategies_values / generate_nash_values).  The
solver matches each question against the generator's own template texts (read with `git show` from the
generation repo history, used only as a parsing grammar, never for answers) to recover the variable values, then
recomputes the game.  Payoff semantics follow the generator: finance payoff = price x units sold; job-market
dominant-strategy payoff = salary increase (the acceptor gets 0 when the other negotiates); job-market Nash payoff
= hires x per-hire margin as written in each of the 5 templates.

Solution concepts: dominance is strict (as find_dom_strategies); pure Nash uses weak best responses (as
find_nash_equilibrium), tolerance 1e-9; payoff/value matching tolerance 1e-6 absolute.
"""
from __future__ import annotations

import ast
import io
import re
import subprocess
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
GEN_REPO = Path(__file__).resolve().parents[3]

REPAIRS = {
    "options_rebuilt": "options replaced by the responding player's action labels (release showed the other "
                       "player's labels, one per matrix cell); stored cell answer translated to that action",
    "answer_from_gen_copy": "part had 0 correct options in the release; answer taken from the generation-repo copy "
                            "(STEER-generation 634bc8f elements/iterated_removal), which the solver confirms",
    "single_option_dropped": "only one distinct option left after collapsing repeated option texts",
    "float_artifact_rounded": "binary floating-point artefact (e.g. 5.890000000000001) rounded to 2 decimals in the "
                              "payoff table / option texts",
}

LAB = r"Action [A-Z]"
UNFILLED = re.compile(r"\{\w+\}|\\+\(\w+\\+\)")  # template variable left in the text
WORD_N = {"two": 2, "three": 3, "four": 4, "five": 5}


def _out(df):
    return pd.DataFrame({"pred": pd.array([pd.NA] * len(df), dtype="Int64"), "status": "unparsed"}, index=df.index)


# ============================================================================================ matrix parsing
_HDR = re.compile(r"\s*(?:Action [A-Z]\s*)+")
_ROW = re.compile(r"\s*(Action [A-Z])\s+(.*)")
_PAIR = re.compile(r"\(\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*\)")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class Table:
    __slots__ = ("rows", "cols", "P")

    def __init__(self, rows, cols, P):
        self.rows, self.cols, self.P = rows, cols, P  # P: (n, m, k) k=2 bimatrix, k=1 single payoff

    @property
    def bimatrix(self):
        return self.P.shape[2] == 2


def parse_tables(text: str) -> list[Table]:
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if _HDR.fullmatch(lines[i]) and not lines[i].strip().startswith("State"):
            cols = re.findall(LAB, lines[i])
            rows, cells = [], []
            j = i + 1
            while j < len(lines) and _ROW.match(lines[j]) and not _HDR.fullmatch(lines[j]):
                m = _ROW.match(lines[j])
                rest = m.group(2)
                prs = _PAIR.findall(rest)
                vals = [(float(a), float(b)) for a, b in prs] if prs else [(float(x),) for x in _NUM.findall(rest)]
                rows.append(m.group(1))
                cells.append(vals)
                j += 1
            if rows and all(len(c) == len(cols) for c in cells) and len({len(v) for c in cells for v in c}) == 1:
                out.append(Table(rows, cols, np.array(cells, dtype=float)))
            else:
                out.append(None)
            i = j
        else:
            i += 1
    return out


def role_of(text):
    m = re.search(r"You are the (row|column) player", text)
    return m.group(1) if m else None


def other(role):
    return {"row": "column", "column": "row"}[role]


def payoff(t: Table, player: str) -> np.ndarray:
    """(n, m) payoffs of `player` ('row'/'column'); single-payoff tables are returned as is."""
    if t.bimatrix:
        return t.P[:, :, 0 if player == "row" else 1]
    return t.P[:, :, 0]


def own_view(t: Table, player: str, U: np.ndarray | None = None):
    """(actions, matrix with the player's actions as rows)."""
    U = payoff(t, player) if U is None else U
    return (t.rows, U) if player == "row" else (t.cols, U.T)


def strictly_dominant(U):
    """indices a such that row a strictly dominates every other row of U."""
    n = U.shape[0]
    return [a for a in range(n) if all((U[a] > U[b] + TOL).all() for b in range(n) if b != a)]


def strictly_dominated(U):
    n = U.shape[0]
    return [a for a in range(n) if any((U[b] > U[a] + TOL).all() for b in range(n) if b != a)]


def pure_ne(A, B):
    """weak pure Nash equilibria (i, j) of the bimatrix game (A row payoffs, B column payoffs)."""
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] >= A[:, j].max() - TOL and B[i, j] >= B[i, :].max() - TOL]


def mixed_ne(A, B):
    """Nash equilibria with at least one player mixing, by equal-size support enumeration (non-degenerate games)."""
    n, m = A.shape
    eqs = []
    for k in range(2, min(n, m) + 1):
        for I in combinations(range(n), k):
            for J in combinations(range(m), k):
                # column mix q on J makes row indifferent on I; row mix p on I makes column indifferent on J
                def solve(M):  # M: k x k payoffs; find prob vector x with M x = v 1, sum x = 1
                    lhs = np.zeros((k + 1, k + 1))
                    lhs[:k, :k] = M
                    lhs[:k, k] = -1
                    lhs[k, :k] = 1
                    rhs = np.zeros(k + 1)
                    rhs[k] = 1
                    try:
                        s = np.linalg.solve(lhs, rhs)
                    except np.linalg.LinAlgError:
                        return None
                    return s[:k], s[k]
                r1 = solve(A[np.ix_(I, J)])
                r2 = solve(B[np.ix_(I, J)].T)
                if r1 is None or r2 is None:
                    continue
                (q, v), (p, w) = r1, r2
                if (q <= 1e-9).any() or (p <= 1e-9).any():
                    continue
                qf = np.zeros(m)
                qf[list(J)] = q
                pf = np.zeros(n)
                pf[list(I)] = p
                if (A @ qf).max() > v + 1e-7 or (pf @ B).max() > w + 1e-7:
                    continue
                eqs.append((pf, qf))
    return eqs


def opt_labels(o: str) -> list[str]:
    return re.findall(LAB, o)


def fnum(s):
    m = _NUM.search(str(s).replace(",", ""))
    return float(m.group(0)) if m else None


def match_one(options, pred_fn):
    """indices of options for which pred_fn(option) is True; collapse identical texts."""
    hits = [k for k, o in enumerate(options) if pred_fn(o)]
    if len({options[k] for k in hits}) == 1:
        return hits[:1], "ok"
    return hits, ("no_match" if not hits else "multi_match")


def _finish(res, idx, hits, st):
    if st == "ok":
        res.at[idx, "pred"] = hits[0]
    res.at[idx, "status"] = st


# ============================================================================================ best_response
_BR = re.compile(r"what action is (your|your opponent's) best response if (?:you play|your opponent plays) (Action [A-Z])\?")


def br_parse(text):
    """(table, responder role, given action label, responder labels, BR label or status)"""
    role = role_of(text)
    m = _BR.search(text)
    ts = parse_tables(text)
    if not (role and m and len(ts) == 1 and ts[0] is not None and ts[0].bimatrix):
        return None
    t = ts[0]
    resp = role if m.group(1) == "your" else other(role)
    g = m.group(2)
    U = payoff(t, resp)
    if resp == "row":
        if g not in t.cols:
            return t, resp, g, t.rows, "ill_posed:label_mismatch"
        v = U[:, t.cols.index(g)]
        labs = t.rows
    else:
        if g not in t.rows:
            return t, resp, g, t.cols, "ill_posed:label_mismatch"
        v = U[t.rows.index(g), :]
        labs = t.cols
    best = np.flatnonzero(v >= v.max() - TOL)
    if len(best) > 1:
        return t, resp, g, labs, "ill_posed:tied_best_response"
    return t, resp, g, labs, labs[best[0]]


def solve_best_response(df):
    res = _out(df)
    for idx, text, opts in zip(df.index, df.question_text, df.options):
        p = br_parse(text)
        if p is None:
            continue
        ans = p[4]
        if ans.startswith("ill_posed"):
            res.at[idx, "status"] = ans
            continue
        hits, st = match_one(list(opts), lambda o: o.strip() == ans)
        _finish(res, idx, hits, st)
    return res


def fix_best_response(df):
    """Rebuild options as the responder's actions; translate the stored cell index into that action."""
    df = df.copy()
    new_opts, new_corr, reps = [], [], []
    for text, opts, corr, rep in zip(df.question_text, df.options, df.correct, df.repair):
        p = br_parse(text)
        if p is None:
            new_opts.append(opts), new_corr.append(corr), reps.append(rep)
            continue
        t, resp, g, labs, _ = p
        c = np.asarray(corr, float)
        w = np.flatnonzero(c == 1)
        stored = None
        if len(w) == 1 and len(opts) == len(t.rows) * len(t.cols):
            i, j = divmod(int(w[0]), len(t.cols))
            stored = t.rows[i] if resp == "row" else t.cols[j]
        new_opts.append(list(labs))
        new_corr.append([1.0 if lab == stored else 0.0 for lab in labs] if stored else [np.nan] * len(labs))
        reps.append(list(rep) + ["options_rebuilt"])
    df["options"], df["correct"], df["repair"] = new_opts, new_corr, reps
    return df


def br_stored_cell_check(df):
    """For the yaml: does the stored cell (i, j) have the given action on the conditioning axis?"""
    n = ok_axis = ok_br = 0
    for text, opts, corr in zip(df.question_text, df.options, df.correct):
        p = br_parse(text)
        if p is None:
            continue
        t, resp, g, labs, ans = p
        w = np.flatnonzero(np.asarray(corr, float) == 1)
        if len(w) != 1:
            continue
        i, j = divmod(int(w[0]), len(t.cols))
        n += 1
        ok_axis += (t.cols[j] == g) if resp == "row" else (t.rows[i] == g)
        ok_br += ((t.rows[i] if resp == "row" else t.cols[j]) == ans)
    return n, ok_axis, ok_br


# ============================================================================================ dominated_strategies
_DS = re.compile(r"(?:what action should (you|your opponent) play\?|which one of (your|your opponent's) actions is dominant\?)")


def solve_dominated_strategies(df):
    res = _out(df)
    for idx, text, opts in zip(df.index, df.question_text, df.options):
        role, m = role_of(text), _DS.search(text)
        ts = parse_tables(text)
        if not (role and m and len(ts) == 1 and ts[0] is not None):
            continue
        t = ts[0]
        subj = "self" if (m.group(1) or m.group(2)) in ("you", "your") else "other"
        chooser = role if subj == "self" else other(role)
        if not t.bimatrix:
            owner = "self" if "find your payoff matrix" in text else "other" if "find your opponent's payoff matrix" in text else None
            if owner != subj:
                res.at[idx, "status"] = "ill_posed:table_owner_mismatch"
                continue
        acts, U = own_view(t, chooser)
        dom = strictly_dominant(U)
        if len(dom) != 1:
            res.at[idx, "status"] = "ill_posed:no_strictly_dominant_action"
            continue
        ans = acts[dom[0]]
        hits, st = match_one(list(opts), lambda o: o.strip() == ans)
        _finish(res, idx, hits, st)
    return res


# ============================================================================================ interpret_games
_IG = re.compile(r"What is (your|your opponent's) payoff if you play (Action [A-Z]) and your opponent plays (Action [A-Z])\?")


def solve_interpret_games(df):
    res = _out(df)
    for idx, text, opts in zip(df.index, df.question_text, df.options):
        role, m = role_of(text), _IG.search(text)
        ts = parse_tables(text)
        if not (role and m and len(ts) == 1 and ts[0] is not None):
            continue
        t = ts[0]
        owner = role if m.group(1) == "your" else other(role)
        if not t.bimatrix:
            tab_owner = "self" if "find your payoff matrix" in text else "other" if "find your opponent's payoff matrix" in text else None
            if tab_owner != ("self" if m.group(1) == "your" else "other"):
                res.at[idx, "status"] = "ill_posed:table_owner_mismatch"
                continue
        mine, theirs = m.group(2), m.group(3)
        r_lab, c_lab = (mine, theirs) if role == "row" else (theirs, mine)
        if r_lab not in t.rows or c_lab not in t.cols:
            res.at[idx, "status"] = "ill_posed:label_mismatch"
            continue
        v = payoff(t, owner)[t.rows.index(r_lab), t.cols.index(c_lab)]
        hits, st = match_one(list(opts), lambda o: fnum(o) is not None and abs(fnum(o) - v) <= 1e-6)
        _finish(res, idx, hits, st)
    return res


_ARTEFACT = re.compile(r"-?\d+\.\d{6,}")


def _round_artefacts(s):
    return _ARTEFACT.sub(lambda m: repr(round(float(m.group(0)), 2)), s)


def fix_float_artefacts(df):
    """Round 5.890000000000001-style float artefacts (payoffs are generated with 2 decimals)."""
    df = df.copy()
    qt = df.question_text.map(_round_artefacts)
    op = df.options.map(lambda o: [_round_artefacts(x) for x in o])
    ch = (qt != df.question_text) | (op.map(tuple) != df.options.map(tuple))
    df["question_text"], df["options"] = qt, op
    df.loc[ch, "repair"] = df.loc[ch, "repair"].map(lambda r: list(r) + ["float_artifact_rounded"])
    return df


def postfix_single_option(df):
    df = df.copy()
    m = df.options.map(len).lt(2) & df["drop"].isna()
    df.loc[m, "drop"] = "single_option"
    df.loc[m, "repair"] = df.loc[m, "repair"].map(lambda r: r + ["single_option_dropped"])
    return df


# ============================================================================================ iterated_removal
_IR0 = re.compile(r"Which one of (your|your opponent's) actions is dominated\?")
_IR1 = re.compile(r"which one of (your|your opponent's) actions is dominated\?|"
                  r"[Ww]hich action should (you|your opponent) not play(?: in the remaining game)?\?")


def ir_solve_base(t0: str, t1: str | None):
    """-> (ans0 label|None, status0, ans1 label|None, status1)"""
    role = role_of(t0)
    m0 = _IR0.search(t0)
    ts = parse_tables(t0)
    if not (role and m0 and len(ts) == 1 and ts[0] is not None and ts[0].bimatrix):
        return None, "unparsed", None, "unparsed"
    t = ts[0]
    p0 = role if m0.group(1) == "your" else other(role)
    acts0, U0 = own_view(t, p0)
    d0 = strictly_dominated(U0)
    if len(d0) != 1:
        st = "ill_posed:no_dominated_action" if not d0 else "ill_posed:multiple_dominated_actions"
        return None, st, None, st
    a0 = acts0[d0[0]]
    if t1 is None:
        return a0, "ok", None, "unparsed"
    m1 = _IR1.search(t1)
    if not m1:
        return a0, "ok", None, "unparsed"
    w = m1.group(1) or m1.group(2)
    p1 = role if w in ("your", "you") else other(role)
    if p1 == p0:
        return a0, "ok", None, "ill_posed:part1_same_player"
    keep = [k for k in range(len(acts0)) if k != d0[0]]
    if p0 == "row":
        red = Table([t.rows[k] for k in keep], t.cols, t.P[keep, :, :])
    else:
        red = Table(t.rows, [t.cols[k] for k in keep], t.P[:, keep, :])
    ts1 = parse_tables(t1)
    if ts1:
        s = ts1[0]
        if s is None or s.rows != red.rows or s.cols != red.cols or not np.allclose(s.P, red.P):
            return a0, "ok", None, "ill_posed:reduced_matrix_mismatch"
    acts1, U1 = own_view(red, p1)
    d1 = strictly_dominated(U1)
    if len(d1) != 1:
        return a0, "ok", None, ("ill_posed:no_dominated_action_after_removal" if not d1
                                else "ill_posed:multiple_dominated_actions_after_removal")
    return a0, "ok", acts1[d1[0]], "ok"


def solve_iterated_removal(df):
    res = _out(df)
    texts = df.set_index(["base_id", "part"]).question_text.to_dict()
    for idx, b, part, opts in zip(df.index, df.base_id, df.part, df.options):
        t0 = texts.get((b, 0))
        if t0 is None:
            continue
        a0, s0, a1, s1 = ir_solve_base(t0, texts.get((b, 1)))
        ans, st = (a0, s0) if part == 0 else (a1, s1)
        if st != "ok":
            res.at[idx, "status"] = st
            continue
        hits, st = match_one(list(opts), lambda o: o.strip() == ans)
        _finish(res, idx, hits, st)
    return res


@lru_cache(maxsize=1)
def _ir_gen_copy():
    """part-1 answers of the generation-repo copy (634bc8f elements/iterated_removal), keyed by question text pair."""
    fr = {}
    for k in ("questions", "options", "answers"):
        b = subprocess.run(["git", "-C", str(GEN_REPO), "show", f"634bc8f:elements/iterated_removal/{k}.pkl"],
                           capture_output=True, check=True).stdout
        fr[k] = pd.read_pickle(io.BytesIO(b))
    q, o, a = fr["questions"], fr["options"], fr["answers"]
    o = o.merge(a, on=["question_id", "option_id"])
    corr = o[o.correct_answer == 1].groupby("question_id").option_text.agg(list)
    q = q.assign(base=q.question_id.str.rsplit("_", n=1).str[0], part=q.question_id.str.rsplit("_", n=1).str[1])
    p0 = q[q.part == "0"].set_index("base").question_text
    p1 = q[q.part == "1"].set_index("base")
    out = {}
    for b, r in p1.iterrows():
        c = corr.get(r.question_id, [])
        if len(c) == 1 and b in p0.index:
            out[(p0[b], r.question_text)] = c[0]
    return out


def fix_iterated_removal(df):
    """Fill the 0-correct part-1 answers from the generation-repo copy (matched on the part-0 + part-1 texts)."""
    df = df.copy()
    gen = _ir_gen_copy()
    t0 = df[df.part == 0].set_index("base_id").question_text
    for i in df.index[(df.part == 1) & (df.n_correct == 0)]:
        key = (t0.get(df.at[i, "base_id"]), df.at[i, "question_text"])
        ans = gen.get(key)
        if ans is None:
            continue
        opts = list(df.at[i, "options"])
        if opts.count(ans) != 1:
            continue
        df.at[i, "correct"] = [1.0 if o == ans else 0.0 for o in opts]
        df.at[i, "repair"] = list(df.at[i, "repair"]) + ["answer_from_gen_copy"]
    return df


# ============================================================================================ pure_nash (matrix)
_PN = re.compile(r"what (action should (you|your opponent) play|is the action pair that you and your opponent play) "
                 r"in a pure Nash equilibrium\?")


def pn_matrix(text, opts):
    role, m = role_of(text), _PN.search(text)
    ts = parse_tables(text)
    if not (role and m and len(ts) == 1 and ts[0] is not None and ts[0].bimatrix):
        return None, "unparsed"
    t = ts[0]
    A, B = payoff(t, "row"), payoff(t, "column")
    ne = pure_ne(A, B)
    if not ne:
        return None, "ill_posed:no_pure_ne"
    if m.group(2) is None:
        target = {(t.rows[i], t.cols[j]) for i, j in ne}

        def key(o):
            ls = opt_labels(o)
            r = [x for x in ls if x in t.rows]
            c = [x for x in ls if x in t.cols]
            return (r[0], c[0]) if len(ls) == 2 and len(r) == 1 and len(c) == 1 else None
    else:
        p = role if m.group(2) == "you" else other(role)
        target = {(t.rows[i] if p == "row" else t.cols[j]) for i, j in ne}

        def key(o):
            ls = opt_labels(o)
            return ls[0] if len(ls) == 1 else None
    hits = [k for k, o in enumerate(opts) if key(o) in target]
    if len(hits) == 1:
        return hits[0], "ok"
    if not hits:
        return None, "no_match"
    return None, "ill_posed:multiple_pure_ne"


# ============================================================================================ word problems
_TPL_SRC = {
    "dominant_strategies": [("634bc8f", "generated_templates/dominant_strategies")],
    "pure_nash": [("634bc8f", "generated_templates/pure_nash"), ("336902f", "generated_templates/pure_nash"),
                  ("b7dd724", "generated_templates/pure_nash")],
}


def _rx(t):
    out, seen, pos = [], set(), 0
    for m in re.finditer(r"\{(\w+)\}", t):
        out.append(re.escape(t[pos:m.start()]))
        v = m.group(1)
        out.append(f"(?P={v})" if v in seen else f"(?P<{v}>.+?)")
        seen.add(v)
        pos = m.end()
    out.append(re.escape(t[pos:]))
    return re.compile("".join(out) + r"\Z", re.S)


@lru_cache(maxsize=4)
def templates(el):
    qs = []
    for sha, path in _TPL_SRC[el]:
        b = subprocess.run(["git", "-C", str(GEN_REPO), "show", f"{sha}:{path}/questions.pkl"],
                           capture_output=True, check=True).stdout
        qs.append(pd.read_pickle(io.BytesIO(b)))
    T = pd.concat(qs).drop_duplicates("question_text").reset_index(drop=True)
    tpl = [(t.split("{")[0], _rx(t)) for t in T.question_text]
    idx = {}
    for k, (pre, _) in enumerate(tpl):
        idx.setdefault(pre[:20], []).append(k)
    return tpl, idx


def match_template(el, text):
    tpl, idx = templates(el)
    cand = idx.get(text[:20], [])
    for k in cand + [k for k in range(len(tpl)) if k not in cand]:
        pre, rx = tpl[k]
        if text.startswith(pre):
            m = rx.match(text)
            if m:
                return m
    return None


def _val(s):
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def player_side(m, name):
    """'a' / 'b' / None: which agent's variables sit in clauses that mention `name` (non-symmetric finance)."""
    text = m.string
    spans = sorted((m.start(g), m.end(g), g) for g in m.re.groupindex)
    votes = {"a": 0, "b": 0}
    prev = 0
    for s, e, g in spans:
        seg = text[prev:s]
        if g[:2] in ("a_", "b_") and name in seg:
            votes[g[0]] += 1
        prev = e
    if votes["a"] == votes["b"]:
        return None
    return "a" if votes["a"] > votes["b"] else "b"


def finance_game(v, side):
    """own payoff matrices U[own_action, other_action] (0 = low price, 1 = high price) in units and revenue
    (revenue None when the template shows no prices), for the asked player and for the other player."""
    if "num_both_high" in v:  # symmetric
        u = np.array([[v["num_both_low"], v["num_low"]], [v["num_high"], v["num_both_high"]]], float)
        own, oth = u, u
    else:
        ua = np.array([[v["a_both_low"], v["a_low_b_high"]], [v["a_high_b_low"], v["a_both_high"]]], float)
        ub = np.array([[v["b_both_low"], v["b_low_a_high"]], [v["b_high_a_low"], v["b_both_high"]]], float)
        own, oth = (ua, ub) if side == "a" else (ub, ua)
    if "price_low" not in v:
        return own, oth, None, None
    price = np.array([[v["price_low"]], [v["price_high"]]])
    return own, oth, own * price, oth * price


_OBJ_UNITS = re.compile(r"unit sales|units sold|\bunits\b|sales volume|number of \w+|customers|\bvolume\b", re.I)
_OBJ_REV = re.compile(r"revenue|profit|earnings|income", re.I)


def objective(text):
    """objective named in the question sentence: 'units', 'revenue' or 'ambiguous' ("sales", "payoff", none)."""
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    q = next((x for x in reversed(sents) if "?" in x), sents[-1])
    if _OBJ_UNITS.search(q):
        return "units"
    if _OBJ_REV.search(q):
        return "revenue"
    return "ambiguous"


def finance_setup(el, text, opts, tags):
    """-> (status, v, games) with games = list of (own, oth) payoff matrices, one per admissible reading of the
    objective: revenue = price x units when prices are shown and the question names revenue/profit; units when it
    names units/volume/customers or shows no prices; both readings when it says "sales"/"payoff"/nothing."""
    p = word_parse(el, text, opts)
    if p is None:
        return ("ill_posed:unfilled_template" if UNFILLED.search(text) else "unparsed"), None, None
    m, v = p
    if "price_low" in v and not v["price_low"] < v["price_high"]:
        return "ill_posed:price_order", v, None
    side = "a"
    if "a_both_high" in v:
        names = {_name_from_option(o) for o in opts}
        side = player_side(m, names.pop()) if len(names) == 1 and None not in names else None
        if side is None:  # options do not name the player: fall back on the metadata tag (a_price_* / b_price_*)
            side = "a" if any(t.startswith("a_price") for t in tags) else "b" if any(
                t.startswith("b_price") for t in tags) else None
        if side is None:
            return "unparsed", v, None
    own, oth, own_rev, oth_rev = finance_game(v, side)
    obj = objective(text)
    if own_rev is None or obj == "units":
        return "ok", v, [(own, oth)]
    if obj == "revenue":
        return "ok", v, [(own_rev, oth_rev)]
    return "ok", v, [(own_rev, oth_rev), (own, oth)]


def price_option(o, v):
    """0 = low, 1 = high, None"""
    nums = [float(x) for x in re.findall(r"\$\s*(-?\d+(?:\.\d+)?)", o.replace(",", ""))]
    hit = {0 for x in nums if abs(x - v["price_low"]) < 1e-9} | {1 for x in nums if abs(x - v["price_high"]) < 1e-9}
    if len(hit) == 1:
        return hit.pop()
    lo = re.search(r"\blow(?:er)?\b|\bsale\b|\breduc|\bdiscount|\bcut\b", o, re.I)
    hi = re.search(r"\bhigh(?:er)?\b", o, re.I)
    if bool(lo) != bool(hi) and not nums:
        return 1 if hi else 0
    return None


def _name_from_option(o):
    m = re.match(r"\s*(.+?)\s+(?:should|will|must|can|would)\b", o)
    return m.group(1).strip().strip("'‘’") if m else None


def _dominant(U):
    d = strictly_dominant(U)
    return d[0] if len(d) == 1 else None


def word_parse(el, text, opts):
    m = match_template(el, text)
    if m is None:
        return None
    v = {k: _val(x) for k, x in m.groupdict().items()}
    if any(x is None for x in v.values()):
        return None
    return m, v


def solve_dominant_strategies(df, return_detail=False):
    res = _out(df)
    det = {}
    for idx, text, opts, tags in zip(df.index, df.question_text, df.options, df.tags):
        opts = list(opts)
        if UNFILLED.search(text):
            res.at[idx, "status"] = "ill_posed:unfilled_template"
            continue
        p = word_parse("dominant_strategies", text, opts)
        if p is None:
            continue
        m, v = p
        if "num_both_high" in v or "a_both_high" in v:
            st, v, games = finance_setup("dominant_strategies", text, opts, list(tags))
            if st != "ok":
                res.at[idx, "status"] = st
                continue
            ds = {_dominant(own) for own, _ in games}
            if len(ds) > 1:
                res.at[idx, "status"] = "ill_posed:objective_ambiguous_units_vs_revenue"
                continue
            d = ds.pop()
            if d is None:
                res.at[idx, "status"] = "ill_posed:no_dominant_strategy"
                continue
            lab = [price_option(o, v) for o in opts]
            hits = [k for k, x in enumerate(lab) if x == d]
        else:  # job market: 0 = accept, 1 = negotiate
            U = np.array([[v["both_accept"], 0.0], [v["one_negotiate"], v["both_negotiate"]]])
            d = _dominant(U)
            if d is None:
                res.at[idx, "status"] = "ill_posed:no_dominant_strategy"
                continue
            lab = [1 if re.search(r"negotiat", o, re.I) and not re.search(r"accept", o, re.I)
                   else 0 if re.search(r"accept", o, re.I) and not re.search(r"negotiat", o, re.I) else None
                   for o in opts]
            hits = [k for k, x in enumerate(lab) if x == d]
        _finish(res, idx, hits, "ok" if len(hits) == 1 else "no_match" if not hits else "multi_match")
    if return_detail:
        return res, det
    return res


def _job_margins(text, v):
    """readings of the per-hire margin (low salary, high salary) in the 5 job-market Nash templates.
    Payoff = hires x margin; hires = n/2 if both offer the same salary, else percent_high / percent_low of n."""
    ph = v["profit"] if "profit" in v else 0.0
    if "with the upper offer yielding an additional" in text:
        # CodeMax template: "the upper offer yielding an additional $p profit per developer" -- either only the high
        # offer earns p, or every hire earns p; both readings are computed and must agree.
        return [(-v["low_loss"], ph - v["high_loss"]), (ph - v["low_loss"], ph - v["high_loss"])]
    return [(ph - v["low_loss"], ph - v["high_loss"])]


def pn_word(text, opts, tags):
    if UNFILLED.search(text):
        return None, "ill_posed:unfilled_template"
    p = word_parse("pure_nash", text, opts)
    if p is None:
        return None, "unparsed"
    m, v = p
    if "num_both_high" in v or "a_both_high" in v:
        st, v, games = finance_setup("pure_nash", text, opts, tags)
        if st != "ok":
            return None, st
        tg = {frozenset(i for i, _ in pure_ne(own, oth.T)) for own, oth in games}  # asked player = row player
        if len(tg) > 1:
            return None, "ill_posed:objective_ambiguous_units_vs_revenue"
        target = set(tg.pop())
        if not target:
            return None, "ill_posed:no_pure_ne"
        if len(target) > 1:
            return None, "ill_posed:multiple_pure_ne"
        lab = [price_option(o, v) for o in opts]
        hits = [k for k, x in enumerate(lab) if x in target]
    else:
        n = v["num_workers"]
        fh, fl = v["percent_high"] / 100, v["percent_low"] / 100
        tgts = []
        for ml, mh in _job_margins(text, v):
            # own payoff U[own, other], 0 = low salary, 1 = high salary
            U = np.array([[n / 2 * ml, n * fl * ml], [n * fh * mh, n / 2 * mh]])
            ne = pure_ne(U, U.T)
            tgts.append(frozenset({("HH" if (i, j) == (1, 1) else "LL" if (i, j) == (0, 0) else "HL")
                                   for i, j in ne} or {"none"}))
        if len(set(tgts)) > 1:
            return None, "ill_posed:ambiguous_payoff_text"
        tgt = tgts[0]

        def cls(o):
            ol = o.lower()
            if "no nash" in ol or "no pure" in ol:
                return "none"
            h, lo = "high" in ol, "low" in ol
            if h and lo:
                return "HL"
            return "HH" if h else "LL" if lo else None
        lab = [cls(o) for o in opts]
        if len(tgt) > 1:
            return None, "ill_posed:multiple_pure_ne"
        hits = [k for k, x in enumerate(lab) if x in tgt]
    if len(hits) == 1:
        return hits[0], "ok"
    return None, "no_match" if not hits else "multi_match"


def solve_pure_nash(df):
    res = _out(df)
    for idx, text, opts, tags in zip(df.index, df.question_text, df.options, df.tags):
        opts = list(opts)
        pred, st = pn_matrix(text, opts) if text.startswith("Below, find the payoff matrix") else pn_word(text, opts, list(tags))
        res.at[idx, "status"] = st
        if st == "ok":
            res.at[idx, "pred"] = pred
    return res


# ============================================================================================ subgame_nash
_SG = re.compile(r"what (actions should you play|action should your opponent play|are the action pairs that you "
                 r"and your opponent play) in each round if the game repeats for (\w+) rounds")


def solve_subgame_nash(df, check_mixed=False):
    res = _out(df)
    for idx, text, opts in zip(df.index, df.question_text, df.options):
        role, m = role_of(text), _SG.search(text)
        ts = parse_tables(text)
        if not (role and m and m.group(2) in WORD_N and len(ts) == 1 and ts[0] is not None and ts[0].bimatrix):
            continue
        t = ts[0]
        n = WORD_N[m.group(2)]
        A, B = payoff(t, "row"), payoff(t, "column")
        ne = pure_ne(A, B)
        if len(ne) != 1:
            res.at[idx, "status"] = "ill_posed:no_pure_stage_ne" if not ne else "ill_posed:multiple_stage_ne"
            continue
        if check_mixed and mixed_ne(A, B):
            res.at[idx, "status"] = "ill_posed:mixed_stage_ne"
            continue
        i, j = ne[0]
        q = m.group(1)
        if q.startswith("are"):   # options name your action first
            exp = ([t.rows[i], t.cols[j]] if role == "row" else [t.cols[j], t.rows[i]]) * n
        else:
            p = role if q.startswith("actions should you") else other(role)
            exp = [t.rows[i] if p == "row" else t.cols[j]] * n
        hits, st = match_one(list(opts), lambda o: opt_labels(o) == exp)
        _finish(res, idx, hits, st)
    return res


# ============================================================================================ bayes_nash
_PROB = re.compile(r"probability of state (\d+) is (-?\d*\.?\d+)", re.I)


def solve_bayes_nash(df):
    res = _out(df)
    for idx, text, opts in zip(df.index, df.question_text, df.options):
        probs = {int(k): float(p) for k, p in _PROB.findall(text)}
        states = [int(s) for s in re.findall(r"^State (\d+):\s*$", text, re.M)]
        ts = parse_tables(text)
        if not (probs and states and len(ts) == len(states) and all(t is not None and t.bimatrix for t in ts)
                and set(states) == set(probs)):
            continue
        if abs(sum(probs.values()) - 1) > 0.011:
            res.at[idx, "status"] = "ill_posed:probabilities_do_not_sum_to_1"
            continue
        t0 = ts[0]
        if any(t.rows != t0.rows or t.cols != t0.cols for t in ts):
            continue
        E = sum(probs[s] * t.P for s, t in zip(states, ts))
        ne = pure_ne(E[:, :, 0], E[:, :, 1])
        if len(ne) != 1:
            res.at[idx, "status"] = "ill_posed:no_pure_bne" if not ne else "ill_posed:multiple_pure_bne"
            continue
        i, j = ne[0]
        hits, st = match_one(list(opts), lambda o: opt_labels(o) == [t0.rows[i], t0.cols[j]])
        _finish(res, idx, hits, st)
    return res


# ============================================================================================ registry
SOLVERS = {
    "best_response": solve_best_response,
    "dominated_strategies": solve_dominated_strategies,
    "interpret_games": solve_interpret_games,
    "iterated_removal": solve_iterated_removal,
    "pure_nash": solve_pure_nash,
    "subgame_nash": solve_subgame_nash,
    "dominant_strategies": solve_dominant_strategies,
    "bayes_nash": solve_bayes_nash,
}

FIXES = {
    "best_response": fix_best_response,
    "iterated_removal": fix_iterated_removal,
    "interpret_games": fix_float_artefacts,
    "dominated_strategies": fix_float_artefacts,
}

POSTFIX = {
    "interpret_games": postfix_single_option,
}

_P = {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed", "unparsed", "no_match", "multi_match"],
      "shuffle": None}
POLICY = {el: dict(_P) for el in SOLVERS}
