"""Solvers and repair hooks for the voting elements (v1-era, no creation code on this machine).

Elements: condorcet_criterion, borda_count, plurality_voting, pareto_sc.

Common structure: a preference profile written as lines "<n> voters voted for X > Y > Z" and candidates
named by single capital letters (options "Candidate X", "Band X", ...).

Partial ballots.  Many profiles rank fewer candidates than there are (condorcet: 3 of 4, borda/plurality:
3 of 4 or 5, pareto_sc type 1: all but one).  Convention used here (the usual reading of truncated
ballots): every ranked candidate is above every unranked one, unranked candidates are tied among themselves.
Where the answer depends on this convention the row is ill_posed:
  condorcet  -- Condorcet winner differs if pairs involving an unranked candidate are ignored instead;
  borda      -- winner / second place differs between the paper's scoring (m-1, ..., 0 over m candidates,
                unranked 0) and the average-rank variant for the unranked, or the ballot-length scoring
                (L-1, ..., 0 over the L ranked candidates) used in some of the GPT explanations;
  pareto_sc  -- with exactly one unranked candidate the completion is unique (it is last), so no ambiguity.
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd

REPAIRS = {
    "options_from_text": "Yes/No options were glued onto the single option text; split into question + Yes/No",
}

BALLOT = re.compile(r"(\d+)\s+voters?\s+voted\s+for\s+([A-Z](?:\s*>\s*[A-Z])*)(?![A-Za-z])")
BALLOT_ANY = re.compile(r"\d*\s*voters?\s+voted\s+for\s+[A-Z](?:\s*>\s*[A-Z])*")
LETTER = re.compile(r"(?<![A-Za-z'])([A-Z])(?='s\b|(?![A-Za-z']))")
LABEL = re.compile(r"(?:^|\s)([A-Z])\s*\.?$")


def _out(index, preds, stats):
    return pd.DataFrame({"pred": pd.array(preds, dtype="Int64"), "status": stats}, index=index)


def parse_ballots(text: str) -> list[tuple[int, tuple[str, ...]]]:
    return [(int(n), tuple(re.split(r"\s*>\s*", b.strip()))) for n, b in BALLOT.findall(text)]


def option_labels(options) -> list[str | None]:
    """Single-letter candidate label of each option ("Candidate X" -> "X"), else None."""
    out = []
    for o in options:
        m = LABEL.search(str(o).strip())
        out.append(m.group(1) if m else None)
    return out


def complete(ballots, cands):
    """Append unranked candidates (as a tied block) -> list of (n, ranked tuple, frozenset unranked)."""
    return [(n, b, frozenset(cands) - set(b)) for n, b in ballots]


def pairwise(ballots, cands, ignore_unranked=False):
    """P[a][b] = number of voters ranking a strictly above b."""
    P = {a: {b: 0 for b in cands} for a in cands}
    for n, b, un in complete(ballots, cands):
        pos = {c: i for i, c in enumerate(b)}
        for x, y in combinations(cands, 2):
            px, py = pos.get(x), pos.get(y)
            if px is None and py is None:
                continue
            if ignore_unranked and (px is None or py is None):
                continue
            if py is None or (px is not None and px < py):
                P[x][y] += n
            else:
                P[y][x] += n
    return P


def condorcet_winner(P, cands):
    w = [a for a in cands if all(P[a][b] > P[b][a] for b in cands if b != a)]
    return w[0] if w else None


def borda_scores(ballots, cands, conv="paper"):
    """conv: 'paper' m-1-i, unranked 0; 'avg' unranked share the leftover points; 'ballot' L-1-i, unranked 0."""
    m = len(cands)
    s = {c: 0.0 for c in cands}
    for n, b, un in complete(ballots, cands):
        L = len(b)
        for i, c in enumerate(b):
            s[c] += n * ((m - 1 - i) if conv != "ballot" else (L - 1 - i))
        if conv == "avg" and un:
            k = len(un)
            for c in un:
                s[c] += n * (k - 1) / 2
    return s


def _cands_from_options(options):
    labs = option_labels(options)
    return labs, [x for x in dict.fromkeys(labs) if x is not None]


# ========================================================================================= plurality_voting
def solve_plurality(df):
    preds, stats = [], []
    for text, opts in zip(df.question_text, df.options):
        ballots = parse_ballots(text)
        labs, cands = _cands_from_options(opts)
        if not ballots or not cands:
            preds.append(None); stats.append("unparsed"); continue
        if any(c not in cands for _, b in ballots for c in b):
            preds.append(None); stats.append("unparsed:ballot_candidate_not_in_options"); continue
        cnt = {c: 0 for c in cands}
        for n, b in ballots:
            cnt[b[0]] += n
        top = max(cnt.values())
        win = [c for c in cands if cnt[c] == top]
        if len(win) > 1:
            preds.append(None); stats.append("ill_posed:tie"); continue
        idx = [i for i, l in enumerate(labs) if l == win[0]]
        preds.append(idx[0] if len(idx) == 1 else None)
        stats.append("ok" if len(idx) == 1 else ("no_match" if not idx else "multi_match"))
    return _out(df.index, preds, stats)


# ============================================================================================== borda_count
# The stored answers follow the paper's scoring (m-1-i over all m candidates, unranked 0): where the paper and
# the ballot-length scoring disagree, stored == paper in 88% and == ballot-length in 1% of rows. Rows where the
# two m-based variants (unranked 0 vs unranked averaged) disagree are ill_posed. BORDA_STRICT additionally
# treats disagreement with the ballot-length scoring as ill_posed (open question for Narun).
BORDA_STRICT = False
SECOND = re.compile(r"second|runner[- ]up", re.I)


def borda_rank(ballots, cands, place, conv):
    """Candidate in `place` (0 = winner, 1 = second) or None if a tie decides that place."""
    s = borda_scores(ballots, cands, conv)
    order = sorted(cands, key=lambda c: -s[c])
    target = order[place] if place < len(order) else None
    if target is None or sum(1 for c in cands if s[c] == s[target]) > 1:
        return None
    return target


def solve_borda(df):
    preds, stats = [], []
    for text, opts, tags in zip(df.question_text, df.options, df.tags):
        ballots = parse_ballots(text)
        labs, cands = _cands_from_options(opts)
        if not ballots or not cands:
            preds.append(None); stats.append("unparsed"); continue
        if any(c not in cands for _, b in ballots for c in b):
            preds.append(None); stats.append("unparsed:ballot_candidate_not_in_options"); continue
        place = 1 if SECOND.search(text) else 0
        tagp = 1 if "second" in list(tags) else (0 if "first" in list(tags) else place)
        if tagp != place:
            preds.append(None); stats.append("unparsed:place_text_tag_conflict"); continue
        w = borda_rank(ballots, cands, place, "paper")
        if w is None:
            preds.append(None); stats.append("ill_posed:tie"); continue
        full = all(len(b) == len(cands) for _, b in ballots)
        if not full:
            alts = {borda_rank(ballots, cands, place, c) for c in (("avg", "ballot") if BORDA_STRICT else ("avg",))}
            if alts != {w}:
                preds.append(None); stats.append("ill_posed:partial_ballot_convention"); continue
        idx = [i for i, l in enumerate(labs) if l == w]
        preds.append(idx[0] if len(idx) == 1 else None)
        stats.append("ok" if len(idx) == 1 else ("no_match" if not idx else "multi_match"))
    return _out(df.index, preds, stats)


# ====================================================================================== condorcet_criterion
# All condorcet profiles rank 3 candidates; with 4 candidates exactly one is unranked per ballot, so the
# completion "unranked = last" is unique (as if full rankings were truncated). Where ignoring the unranked
# candidate's pairs would change the winner (87k rows), the stored answer follows "last" in 77% vs 8%.
# CONDORCET_STRICT=True makes those rows ill_posed instead (open question for Narun).
CONDORCET_STRICT = False
_ARTICLES = {"A", "I"}


def known_candidates(text: str, ballots, options=()) -> list[str]:
    """Candidate letters: those in ballots, single capitals listed before the first ballot line, and
    trailing labels of short options ("Playground A")."""
    head = BALLOT.split(text, maxsplit=1)[0]
    c = dict.fromkeys(x for _, b in ballots for x in b)
    for o in options:
        m = LABEL.search(str(o).strip())
        if m and len(str(o)) <= 40:
            c.setdefault(m.group(1))
    for x in LETTER.findall(head):
        if x not in _ARTICLES or re.search(rf"(?:\b[A-Z][a-z]+ |, | and | or ){x}(?![A-Za-z'])", head):
            c.setdefault(x)
    return list(c)


def statement_labels(options, cands) -> list[str | None]:
    """Candidate referenced by each option: trailing label, else the unique known candidate letter in it."""
    out = []
    for o in options:
        o = str(o).strip()
        m = LABEL.search(o)
        if m and m.group(1) in cands:
            out.append(m.group(1)); continue
        hits = {x for x in LETTER.findall(o) if x in cands}
        out.append(hits.pop() if len(hits) == 1 else (None if not hits else "?"))
    return out


def solve_condorcet(df):
    preds, stats = [], []
    for text, opts in zip(df.question_text, df.options):
        ballots = parse_ballots(text)
        if not ballots:
            preds.append(None); stats.append("unparsed"); continue
        cands = known_candidates(text, ballots)
        labs = statement_labels(opts, cands)
        if "?" in labs:
            preds.append(None); stats.append("unparsed:option_names_several_candidates"); continue
        if len(cands) < 2 or len(set(l for l in labs if l)) != len(cands):
            preds.append(None); stats.append("unparsed:candidates_vs_options"); continue
        P = pairwise(ballots, cands)
        w = condorcet_winner(P, cands)
        n_un = max(len(cands) - len(b) for _, b in ballots)
        if n_un > 1 or (CONDORCET_STRICT and n_un == 1):
            w2 = condorcet_winner(pairwise(ballots, cands, ignore_unranked=True), cands)
            if w2 != w:
                preds.append(None); stats.append("ill_posed:partial_ballot_convention"); continue
        if w is None and any(P[a][b] == P[b][a] for a, b in combinations(cands, 2)) and any(
                all(P[a][b] >= P[b][a] for b in cands if b != a) for a in cands):
            preds.append(None); stats.append("ill_posed:pairwise_tie_weak_winner"); continue
        if w is not None:
            idx = [i for i, l in enumerate(labs) if l == w]
        else:  # the "no Condorcet winner / tie" option: the only non-empty option naming no candidate
            idx = [i for i, (l, o) in enumerate(zip(labs, opts)) if l is None and str(o).strip()]
        preds.append(idx[0] if len(idx) == 1 else None)
        stats.append("ok" if len(idx) == 1 else ("no_match" if not idx else "multi_match"))
    return _out(df.index, preds, stats)


CONDORCET_Q = "\nWhich option should be chosen according to the Condorcet criterion?"


def fix_condorcet(df):
    """(1) 1,500 rows (pos 75060-367559, 500 per type) have the ballot lines spilled into the options, voter
    counts truncated/missing and no question sentence: not repairable -> drop.
    (2) ~291k rows end right after the ballot lines (no question sentence, 'Condorcet' not mentioned
    anywhere in 99% of them): append a neutral question sentence (text_fixed)."""
    spilled = df.options.map(lambda o: any(BALLOT_ANY.search(str(x)) for x in o))
    df.loc[spilled & df["drop"].isna(), "drop"] = "ballots_in_options"
    noq = df.question_text.str.contains(r"voted for [A-Z](?:\s*>\s*[A-Z])*\s*$", regex=True) & ~spilled
    df.loc[noq, "question_text"] = df.loc[noq, "question_text"].str.rstrip() + CONDORCET_Q
    df.loc[noq, "repair"] = df.loc[noq, "repair"].map(lambda r: r + ["text_fixed"])
    return df


# =============================================================================================== pareto_sc
# Profiles: 2 or 3 candidates; type 2 = full rankings, type 1 = rankings with the last candidate dropped
# (exactly one unranked per ballot -> unique completion). Every profile has a unanimous top candidate.
# Question forms (by tail = text after the last ballot line, and option shape):
#   explicit Yes/No  "S was selected. Does this satisfy the Pareto Efficiency axiom?"  -> Yes iff S undominated
#   implicit Yes/No  "S was selected. Would T have been better / satisfied more ...?"   -> Yes iff T dominates S,
#                    No iff S dominates T (else ill_posed:not_pareto_comparable); reversed wording
#                    ("Is this outcome superior to ...", "Was this the best choice?") flips; double questions
#                    ("Was this optimal, or would T ...?") are ill_posed.
#   candidate options "which X should be chosen?" -> the unique Pareto-undominated candidate.
# Everything else is dropped (no question sentence, hypothetical-scenario options, Yes/No options with a
# rationale that ignores the profile).
YESNO = re.compile(r"^(?:[AB]\)\s*)?(Yes|No)[.!]?$")
PARETO_KW = re.compile(r"pareto", re.I)
# Decision 2026-10-02: Yes/No questions that do not ask about Pareto efficiency (the implicit wording, e.g.
# "would more people have been happier if T were chosen?") test majority preference and are dropped. The Yes/No
# Pareto questions and the "which option should be chosen" questions (unanimous preference) are kept.
DROP_NON_PARETO_YES_NO = True
# polarity of implicit Yes/No questions (all 309 distinct question sentences were checked by hand):
# with an alternative T named: Yes = "T would have been better", except these reversed wordings
REV_ALT = re.compile(r"superior to having|better decision than|less popular", re.I)
# with no alternative named: Yes = "the selected S is the best" for these, else Yes = "another would be better"
REV_NOALT = re.compile(r"optimal|preferred choice|preferred outcome|highest overall|\bbest\b|consistent with", re.I)
DOUBLE = re.compile(r",?\s+or\s+(?:would|if|should|was|is)\b", re.I)


def _tail(text: str) -> tuple[str, str]:
    parts = BALLOT_ANY.split(text)
    return parts[0], (parts[-1].strip() if len(parts) > 1 else "")


def _last_q(tail: str) -> str:
    sents = [x.strip() for x in re.split(r"(?<=[.?!])\s+", tail) if x.strip()]
    return sents[-1] if sents else ""


def dominates(ballots, cands, x, y) -> bool:
    for n, b, un in complete(ballots, cands):
        if n == 0:
            continue
        px, py = (b.index(x) if x in b else None), (b.index(y) if y in b else None)
        if px is None or (py is not None and py < px):
            return False
    return True


def pareto_profile(text, options=()):
    ballots = parse_ballots(text)
    if not ballots:
        return None, None, "unparsed:no_ballots"
    cands = known_candidates(text, ballots, options)
    L = max(len(b) for _, b in ballots)
    if len(cands) - min(len(b) for _, b in ballots) > 1:
        return None, None, "ill_posed:several_unranked"
    if len(cands) < 2 or len(cands) > L + 1:
        return None, None, "unparsed:candidates"
    return ballots, cands, None


def solve_pareto(df):
    preds, stats = [], []
    for text, opts, tags in zip(df.question_text, df.options, df.tags):
        def put(p, st):
            preds.append(p); stats.append(st)
        opts = [str(o) for o in opts]
        if any(BALLOT_ANY.search(o) for o in opts):
            put(None, "ill_posed:ballots_in_options"); continue
        ballots, cands, err = pareto_profile(text, opts)
        if err:
            put(None, err); continue
        head, tail = _tail(text)
        undom = [c for c in cands if not any(dominates(ballots, cands, o, c) for o in cands if o != c)]
        nonempty = [o for o in opts if o.strip()]
        yn = [YESNO.match(o.strip()) for o in nonempty]
        if len(nonempty) == 2 and all(yn):
            words = [m.group(1) for m in yn]
            if sorted(words) != ["No", "Yes"]:
                put(None, "unparsed:yes_no_options"); continue
            if not tail:
                put(None, "ill_posed:no_question"); continue
            letters = [x for x in LETTER.findall(tail) if x in cands]
            if not letters:
                put(None, "unparsed:no_selected_candidate"); continue
            S = letters[0]
            q = _last_q(tail)
            if re.match(r"If\b.*\bhad been (?:selected|chosen) instead", q) and len(set(letters)) == 2:
                S = [x for x in letters if x != S][0]  # "If T had been selected instead, ... over S?"
            qlet = [x for x in LETTER.findall(q) if x in cands and x != S]
            if DROP_NON_PARETO_YES_NO and not PARETO_KW.search(tail):
                # "Would more people have been happier if T were chosen?" asks a majority question, not Pareto
                # efficiency (Narun, 2026-10-02): these Yes/No rows are dropped from the element
                put(None, "ill_posed:majority_wording"); continue
            if PARETO_KW.search(tail):
                if set(letters) != {S}:
                    put(None, "unparsed:pareto_question_letters"); continue
                ans = "Yes" if S in undom else "No"
            else:
                if DOUBLE.search(q):
                    put(None, "ill_posed:double_question"); continue
                if len(set(qlet)) > 1:
                    put(None, "unparsed:several_alternatives"); continue
                others = [c for c in cands if c != S]
                if qlet:
                    T, rev = qlet[0], bool(REV_ALT.search(q))
                elif REV_NOALT.search(q):
                    T, rev = None, True
                else:  # alternative named earlier in the tail ("... wonder if O might be better. Could they be correct?")
                    tl = sorted({x for x in letters if x != S})
                    T, rev = (tl[0] if len(tl) == 1 else None), False
                if T is not None:
                    better = dominates(ballots, cands, T, S)
                    worse = dominates(ballots, cands, S, T)
                elif rev:  # "Was this (S) the best choice?"
                    worse = all(dominates(ballots, cands, S, a) for a in others)
                    better = any(dominates(ballots, cands, a, S) for a in others)
                else:      # "Would a different option have been better?"
                    better = any(dominates(ballots, cands, a, S) for a in others)
                    worse = all(dominates(ballots, cands, S, a) for a in others)
                if better == worse:
                    put(None, "ill_posed:not_pareto_comparable"); continue
                ans = ("No" if better else "Yes") if rev else ("Yes" if better else "No")
            put([i for i, o in enumerate(opts) if YESNO.match(o.strip()) and YESNO.match(o.strip()).group(1) == ans][0],
                "ok")
            continue
        labs = statement_labels(opts, cands)
        short = all(len(o) <= 40 for o in nonempty)
        short = short and not any(re.match(r"(?:[AB]\)\s*)?(?:Yes|No)\b", o.strip()) for o in nonempty)
        if len(nonempty) >= 2 and short and all(l not in (None, "?") for l, o in zip(labs, opts) if o.strip()):
            if not tail:
                put(None, "ill_posed:no_question"); continue
            cand_und = [c for c in undom if c in labs]
            if len(cand_und) != 1:
                put(None, "ill_posed:several_pareto_optimal" if cand_und else "no_match"); continue
            idx = [i for i, l in enumerate(labs) if l == cand_und[0]]
            put(idx[0] if len(idx) == 1 else None, "ok" if len(idx) == 1 else "multi_match")
            continue
        if not tail:
            put(None, "ill_posed:options_are_scenarios_no_question"); continue
        put(None, "ill_posed:rationale_options")
    return _out(df.index, preds, stats)


PARETO_Q_EX = "\nAccording to the Pareto Efficiency axiom, which option should be chosen?"
PARETO_Q_IM = "\nWhich option should be chosen?"
GLUED_YN = re.compile(r"^(.*\?)\s+Yes\s+No\s*$", re.S)


def fix_pareto(df):
    """(1) 10,000 rows whose question was cut after the preamble and whose options are ['', '00 voters voted
    for ...'] -> drop (ballots_in_options). (2) 2,000 one-option rows whose single option is
    '<question>?  Yes   No': the question is moved to the text and the options become Yes/No
    (options_from_text; stored answer discarded, the solver fills it)."""
    spilled = df.options.map(lambda o: any(BALLOT_ANY.search(str(x)) for x in o))
    df.loc[spilled & df["drop"].isna(), "drop"] = "ballots_in_options"
    # candidate options but no question sentence: append one
    add = []
    for i, text, opts, tags in zip(df.index, df.question_text, df.options, df.tags_raw):
        if spilled[i] or _tail(text)[1]:
            continue
        ballots, cands, err = pareto_profile(text, opts)
        if err:
            continue
        opts = [str(o) for o in opts if str(o).strip()]
        labs = statement_labels(opts, cands)
        if len(opts) >= 2 and all(len(o) <= 40 for o in opts) and all(l not in (None, "?") for l in labs) \
                and len(set(labs)) == len(labs) and not any(YESNO.match(o.strip()) for o in opts):
            add.append((i, str(tags or "").startswith("explicit")))
    for i, ex in add:
        df.at[i, "question_text"] = df.at[i, "question_text"].rstrip() + (PARETO_Q_EX if ex else PARETO_Q_IM)
        df.at[i, "repair"] = df.at[i, "repair"] + ["text_fixed"]
    one = df.options.map(lambda o: len(o) == 1 and bool(GLUED_YN.match(str(o[0]).strip())))
    for i in df.index[one]:
        q = GLUED_YN.match(str(df.at[i, "options"][0]).strip()).group(1)
        df.at[i, "question_text"] = df.at[i, "question_text"].rstrip() + "\n" + q
        df.at[i, "options"] = ["Yes", "No"]
        df.at[i, "correct"] = [float("nan"), float("nan")]
        df.at[i, "repair"] = df.at[i, "repair"] + ["options_from_text"]
    return df


SOLVERS = {
    "plurality_voting": solve_plurality,
    "borda_count": solve_borda,
    "condorcet_criterion": solve_condorcet,
    "pareto_sc": solve_pareto,
}
FIXES = {"condorcet_criterion": fix_condorcet, "pareto_sc": fix_pareto}
_P = {"grading": "answer", "authoritative": True,
      "drop_status": ["ill_posed", "unparsed", "no_match", "multi_match"], "shuffle": None}
# Authoritative: condorcet disagreements (19% of rows) were checked with an independent pairwise count on
# full-ballot rows (coordinator, 40k sample): independent == solver 3,221/3,332, == stored 35/3,332.
# borda: hand-checked 'second place' disagreements, stored answers wrong.
POLICY = {el: dict(_P) for el in SOLVERS}
