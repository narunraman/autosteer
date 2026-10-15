"""Phase 2 solvers/hooks for the logic and theory-of-mind elements (+ level_k).

Elements: categorical_syllogism, conditional_syllogism, equiv_contrapositive, false_belief,
first_order_false_belief, second_order_false_belief, level_k.

All solvers work from question/option text only (no option positions, no stored answers).
The logic questions are free-form LLM generations, so the parsers are deliberately conservative: a row
gets status "ok" only if every option could be classified; anything else is "unparsed"/"no_match".
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================ shared text utilities
_IRREG = {"lost": "lose", "won": "win", "ran": "run", "went": "go", "gone": "go", "got": "get", "saw": "see",
          "seen": "see", "took": "take", "taken": "take", "made": "make", "found": "find", "did": "do",
          "done": "do", "has": "have", "had": "have", "was": "be", "were": "be", "is": "be", "are": "be",
          "am": "be", "been": "be", "being": "be", "grew": "grow", "grown": "grow", "fell": "fall",
          "fallen": "fall", "paid": "pay", "bought": "buy", "sold": "sell", "kept": "keep", "left": "leave",
          "ate": "eat", "eaten": "eat", "children": "child", "people": "person", "men": "man", "women": "woman",
          "cacti": "cactus", "mice": "mouse", "teeth": "tooth", "feet": "foot", "geese": "goose",
          "leaves": "leaf", "wolves": "wolf", "knives": "knife", "lives": "life", "wives": "wife",
          "fish": "fish", "sheep": "sheep", "species": "species", "series": "series", "bus": "bus",
          "does": "do", "goes": "go", "gets": "get", "caught": "catch", "led": "lead", "met": "meet",
          "drank": "drink", "drunk": "drink", "slept": "sleep", "felt": "feel", "thought": "think", "wrote": "write",
          "written": "write", "spent": "spend", "began": "begin", "begun": "begin", "chose": "choose",
          "chosen": "choose", "forgot": "forget", "forgotten": "forget", "gave": "give", "given": "give",
          "knew": "know", "known": "know", "rose": "rise", "risen": "rise", "woke": "wake", "woken": "wake",
          "brought": "bring", "sent": "send", "built": "build", "held": "hold", "heard": "hear", "told": "tell",
          "understood": "understand", "became": "become", "came": "come"}
_NEG = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor"}
_STOP = set("""a an the then they them their theirs it its he she his her hers him you your yours we our us i me my
one someone somebody something that this these those which who whom whose to of in at for by with as and or
if when whenever there so be have do will would shall should may might must also still just
been being also too very certain certainly given same such any each all every doe goe definitely surely""".split())


def _pre(t: str) -> str:
    t = str(t).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\bcannot\b", "can not", t, flags=re.I)
    t = re.sub(r"\bwon't\b", "will not", t, flags=re.I)
    t = re.sub(r"\bcan't\b", "can not", t, flags=re.I)
    t = re.sub(r"n't\b", " not", t, flags=re.I)
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", t)  # join hyphenated compounds (no-parking, non-steroidal)
    return t


def stem(w: str) -> str:
    w = w.lower()
    if w in _IRREG:
        w = _IRREG[w]
        if w in ("be", "have", "do", "go", "bus", "species", "series"):
            return w
    for suf in ("ies", "ing", "ed", "es", "s", "e", "y"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w


def toks(t: str) -> list[str]:
    return re.findall(r"[a-z0-9$%]+", _pre(t).lower())


def content(t: str) -> tuple[set[str], int]:
    """(stemmed content-word set, negation parity) of a clause."""
    ws = toks(t)
    neg = sum(w in _NEG for w in ws) % 2
    out = set()
    for w in ws:
        if w in _NEG or w in _STOP:
            continue
        sw = stem(w)
        if sw not in _STOP:
            out.add(sw)
    return out, neg


def _result(index, preds, statuses) -> pd.DataFrame:
    return pd.DataFrame({"pred": pd.array(preds, dtype="Int64"), "status": statuses}, index=index)


def _clean_opts(opts) -> list[str]:
    return [str(o) for o in list(opts)]


# ============================================================================ equiv_contrapositive
_IF = r"(?:if|when|whenever)"
_IF_PAIR = [  # (regex, swapped): swapped means the regex's q-group is the antecedent
    (re.compile(r"^\s*" + _IF + r"\s+(?P<p>.+?)\s*,?\s*then\s+(?P<q>.+?)\s*$", re.I), False),
    (re.compile(r"^\s*" + _IF + r"\s+(?P<p>.+?),\s*(?P<q>.+?)\s*$", re.I), False),
    (re.compile(r"^\s*(?P<p>.+?)\s*,?\s+only if\s+(?P<q>.+?)\s*$", re.I), False),
    (re.compile(r"^\s*" + _IF + r"\s+(?P<p>.+?)\s+(?P<q>(?:they|he|she|it|you|we|then|there)\b.+?)\s*$", re.I), False),
    (re.compile(r"^\s*(?P<q>[^,]+?)\s*,?\s+(?:if|when|whenever)\s+(?P<p>.+?)\s*$", re.I), False),
]


def split_if(s: str):
    s = s.strip().strip("\"'").strip()
    s = re.sub(r"[.!]+$", "", s).strip()
    for rx, _ in _IF_PAIR:
        m = rx.match(s)
        if m:
            return m.group("p").strip(" ,"), m.group("q").strip(" ,")
    return None


def statement_of(q: str):
    """Extract (P, Q) of the conditional statement in an equiv_contrapositive question."""
    t = _pre_keep(q)
    head = re.split(r"\s*(?:Which of the following|Which one of the following|The logically|The contrapositive|"
                    r"What is the|Which statement|Select the|Choose the|Identify the)", t, maxsplit=1)[0]
    quoted = re.findall(r"\"([^\"]*\bif\b[^\"]*)\"", head, flags=re.I)
    quoted += re.findall(r"(?:^|\s)'([^']*\bif\b[^']*)'(?:\s|$|[.,:])", head, flags=re.I)
    for c in quoted[::-1] + [head]:
        c = c.strip()
        m = re.search(r"([^.:;!?\"]*?)\s+only if\s+([^.;!?\"]+)", c, flags=re.I)
        if m:
            p = re.sub(r"^.*\b(?:that|thinks|believes|says|claims|states)\s+", "", m.group(1).strip(), flags=re.I)
            return p.strip(" ,"), m.group(2).strip(" ,")
        m = re.search(r"\b(?:if|whenever|when)\b(.*)$", c, flags=re.I | re.S)
        if m and (m.start() == 0 or c[:m.start()].rstrip().endswith((":", ",", "that", "\"", "'")) or
                  re.search(r"\b(?:that|thinks|believes|says|claims|states|notes|rule|reads)\s*[:,]?\s*$", c[:m.start()], re.I)):
            r = split_if("if " + m.group(1).split('"')[0])
            if r:
                return r
        # "<speaker> <Q> if <P>." (consequent first)
        for sent in re.split(r"(?<=[.!?])\s+", c):
            m = re.match(r"^\s*(?P<q>[^,]+?)\s+(?:if|when|whenever)\s+(?P<p>[^.]+?)[.!]?\s*$", sent, re.I)
            if m and not re.match(r"^\s*(?:if|when|whenever)\b", sent, re.I):
                return m.group("p").strip(" ,"), m.group("q").strip(" ,")
    return None


def _pre_keep(t: str) -> str:
    return str(t).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def classify_clause(c: str, P, Q, ctx: set | frozenset = frozenset(), full: bool = False):
    """Return ('P'|'Q', neg) if clause c is (the negation of) P or Q with no extra content, else None.

    ctx: words of the question outside the statement (speaker names etc.) that may appear as subjects."""
    W, n = content(c)
    (WP, nP), (WQ, nQ) = P, Q
    W = W - (ctx - WP - WQ)
    DP, DQ = WP - WQ, WQ - WP
    extra = W - WP - WQ
    if extra or not W:
        return None
    covP = len(W & DP) / len(DP) if DP else 0.0
    covQ = len(W & DQ) / len(DQ) if DQ else 0.0
    if full:  # every distinctive (non-subject) word of the matched clause must be present
        DPs, DQs = DP - ctx, DQ - ctx
        fP = bool(DPs) and DPs <= W and not (W & DQs)
        fQ = bool(DQs) and DQs <= W and not (W & DPs)
        if fP != fQ:
            return ("P", (n + nP) % 2) if fP else ("Q", (n + nQ) % 2)
        return None
    if covP >= 0.5 and covQ <= 0.5 and covP > covQ:
        return "P", (n + nP) % 2
    if covQ >= 0.5 and covP <= 0.5 and covQ > covP:
        return "Q", (n + nQ) % 2
    return None


_MODAL = re.compile(r"\b(?:may|might|could|possibly|perhaps)\b", re.I)


def solve_contrapositive(df: pd.DataFrame) -> pd.DataFrame:
    preds, sts = [], []
    for q, opts in zip(df.question_text, df.options):
        opts = _clean_opts(opts)
        st = statement_of(q)
        if st is None:
            preds.append(pd.NA); sts.append("unparsed"); continue
        P, Q = content(st[0]), content(st[1])
        if not (P[0] - Q[0]) or not (Q[0] - P[0]) or _MODAL.search(st[1]) or _MODAL.search(st[0]):
            preds.append(pd.NA); sts.append("unparsed"); continue
        ctx = content(q.replace(st[0], " ").replace(st[1], " "))[0]
        cls = []
        for o in opts:
            sp = split_if(o) if o.strip() else None
            if sp is None:
                cls.append(None); continue
            a, b = classify_clause(sp[0], P, Q, ctx), classify_clause(sp[1], P, Q, ctx)
            cls.append((a, b) if a and b else None)
        good = [i for i, c in enumerate(cls) if c == (("Q", 1), ("P", 1))]
        if len(good) == 1:
            preds.append(good[0]); sts.append("ok"); continue
        if len(good) > 1:
            preds.append(pd.NA)
            sts.append("ill_posed:two_contrapositives" if len({opts[i] for i in good}) > 1 else "multi_match")
            continue
        preds.append(pd.NA)
        # every option is a literal +-P/+-Q conditional but none is the contrapositive
        sts.append("ill_posed:no_contrapositive_option" if all(c is not None for c in cls) else "unparsed")
    return _result(df.index, preds, sts)


# ============================================================================ conditional_syllogism
_CBD = re.compile(r"cannot be determined|can not be determined|can't be determined|not enough information|"
                  r"undetermined|cannot be concluded|insufficient information|none of the above|may or may not|"
                  r"might or might not|could or could not|not necessarily", re.I)
_POSS = re.compile(r"\b(?:may|might|could)\s+(?:or\s+(?:may|might|could)\s+not\s+)?be\b|\bpossibly\b|\bperhaps\b", re.I)
_UNIV = re.compile(r"^\s*(?:all|every|everyone|everybody|everything|only|no|any|anyone|anything|none|some|each|"
                   r"whoever|whatever|nobody)\b", re.I)


def _names(q: str) -> set[str]:
    """Capitalised tokens not at a sentence start (proper names), stemmed."""
    out = set()
    for m in re.finditer(r"(?<![.!?:]\s)(?<!^)\b([A-Z][a-z]+)\b", _pre(q)):
        out.add(stem(m.group(1)))
    for m in re.finditer(r"(?:^|[.!?]\s+)([A-Z][a-z]+)\b", _pre(q)):
        w = m.group(1).lower()
        if w not in _STOP and w not in ("if", "all", "every", "some", "no", "solely", "which", "when", "whenever"):
            out.add(stem(w))
    return out


def _premise_restated(o: str, P, Q, subj: set = frozenset()) -> bool:
    """Universal option that restates 'if P then Q' (e.g. 'All bakers know how to bake bread')."""
    if re.match(r"^\s*only\b", o, re.I) or re.search(r"\bonly\b", o, re.I):
        return False
    W, n = content(o)
    (WP, nP), (WQ, nQ) = P, Q
    DP, DQ = WP - WQ - subj, WQ - WP - subj
    if not DP or not DQ or (W - WP - WQ - subj):
        return False
    if len(W & DP) / len(DP) < 0.5 or len(W & DQ) / len(DQ) < 0.5:
        return False
    ws = [stem(w) for w in toks(o)]
    ip = min((i for i, w in enumerate(ws) if w in DP), default=None)
    iq = min((i for i, w in enumerate(ws) if w in DQ), default=None)
    neg_q = sum(w in _NEG for w in toks(o)) % 2
    return ip is not None and iq is not None and ip < iq and neg_q == nQ and not re.match(r"^\s*no\b", o, re.I)


def parse_conditional(q: str):
    """(P, Q, fact) strings of a conditional syllogism, or None."""
    t = _pre_keep(q)
    head = re.split(r"\s*(?:Solely from|Based solely|Based on|From the information|Which of the following|Using only)",
                    t, maxsplit=1)[0]
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", head.strip()) if x.strip()]
    if len(sents) != 2:
        return None
    st = split_if(sents[0])
    if st is None:
        return None
    return st[0], st[1], re.sub(r"[.!]+$", "", sents[1]).strip()


def solve_conditional(df: pd.DataFrame) -> pd.DataFrame:
    preds, sts = [], []
    for q, opts in zip(df.question_text, df.options):
        opts = _clean_opts(opts)
        pc = parse_conditional(q)
        if pc is None:
            preds.append(pd.NA); sts.append("unparsed"); continue
        P, Q = content(pc[0]), content(pc[1])
        if not (P[0] - Q[0]) or not (Q[0] - P[0]) or _MODAL.search(pc[1]):
            preds.append(pd.NA); sts.append("unparsed"); continue
        m = re.match(r"^\s*(?:a|an|the|someone|somebody|one)\s+(\w+)\b", pc[0], re.I)
        subj = {stem(m.group(1))} if m and m.group(1).lower() not in ("is", "has", "can") else set()
        if re.match(r"^\s*(?:someone|somebody|one)\b", pc[0], re.I):
            subj |= {"person"}
        ctx = (_names(q) - P[0] - Q[0]) | (P[0] & Q[0]) | subj
        f = classify_clause(pc[2], P, Q, ctx, full=True) if not _UNIV.match(pc[2]) else None
        if f is None:
            preds.append(pd.NA); sts.append("unparsed"); continue
        concl = {("P", 0): ("Q", 0), ("Q", 1): ("P", 1)}.get(f)  # modus ponens / tollens; AC, DA -> None
        cbd = [i for i, o in enumerate(opts) if _CBD.search(o) or _POSS.search(o)]
        ent, unk = [], []
        for i, o in enumerate(opts):
            if i in cbd or not o.strip():
                continue
            if _UNIV.match(o) or re.search(r"\b(?:means|implies|always|only)\b", o, re.I):
                if _premise_restated(o, P, Q, subj):
                    ent.append(i)
                continue
            if re.search(r"\b(?:necessary|required|requirement|therefore|so|because|thus|hence|or)\b", o, re.I):
                unk.append(i)
                continue
            c = classify_clause(o, P, Q, ctx, full=True)
            if c is None:
                unk.append(i)
            elif c == f or (concl is not None and c == concl):
                ent.append(i)
        several = len(ent) > 1 and len({opts[i] for i in ent}) > 1
        if several:
            preds.append(pd.NA); sts.append("ill_posed:several_options_follow")
        elif ent:
            preds.append(ent[0]); sts.append("ok" if len(ent) == 1 else "multi_match")
        elif concl is not None:
            preds.append(pd.NA); sts.append("unparsed" if unk else "no_match")
        elif len(cbd) == 1:
            preds.append(cbd[0]); sts.append("ok")
        elif not cbd and not unk:
            preds.append(pd.NA); sts.append("ill_posed:no_valid_option")
        else:
            preds.append(pd.NA); sts.append("unparsed")
    return _result(df.index, preds, sts)


# ============================================================================ categorical_syllogism
# Sentences are parsed into (kind, S, P) with kind in A/E/I/O and terms as frozensets of stemmed content words;
# singular subjects ("Emily", "the number 6") become individual terms. Entailment is decided by enumerating all
# Venn models over the (<= 3) premise terms. Traditional existential import: every premise term is non-empty.
_GENERIC = {"person", "individual", "anyone", "those", "thing", "object", "creature", "animal", "entit", "one",
            "somebod", "someone", "everyone", "everybod", "member", "type", "kind", "being", "item", "who"}
_AUX = r"(?:are|is|were|was|have|has|had|can|could|do|does|did|will|would|must|should|may|might|need|needs|" \
       r"require|requires|belong|belongs|live|lives|love|loves|like|likes|eat|eats|fly|flies|swim|swims|" \
       r"fade|fades|grow|grows|make|makes|contain|contains|own|owns|work|works|play|plays|know|knows|" \
       r"enjoy|enjoys|wear|wears|use|uses|produce|produces|lay|lays|purr|purrs|bark|barks|go|goes)"


def _term(words: str) -> frozenset | None:
    ws = [w for w in toks(words) if w not in _NEG]
    st = [stem(w) for w in ws if w not in _STOP and stem(w) not in _STOP]
    core = [w for w in st if w not in _GENERIC]
    t = frozenset(core or st)
    return t or None


def parse_cat(sent: str):
    """-> (kind, S, P) or None. S may be ('IND', name) for singular subjects."""
    s = _pre(sent).strip().rstrip(".!").strip()
    s = re.sub(r"\s+", " ", s)
    low = s.lower()
    kind = None
    m = re.match(r"^(not all|not every)\s+(.*)$", low)
    if m:
        kind, rest = "O", m.group(2)
    else:
        m = re.match(r"^(all|every|each|any|no|none of the|some|only)\s+(.*)$", low)
        if m:
            q, rest = m.group(1), m.group(2)
            kind = {"all": "A", "every": "A", "each": "A", "any": "A", "no": "E", "none of the": "E",
                    "some": "I", "only": "ONLY"}[q]
        else:
            rest = low
    # subject / predicate split at the first auxiliary / verb outside a relative clause
    rel = re.match(r"^(.*?\b(?:who|that|which|with|without)\b\s+\S+(?:\s+\S+){0,4}?)\s+(" + _AUX + r")\b\s*(.*)$", rest)
    m2 = rel or re.match(r"^(.*?)\s+(" + _AUX + r")\b\s*(.*)$", rest)
    if not m2:
        return None
    subj, verb, pred = m2.group(1), m2.group(2), m2.group(3)
    neg = bool(re.match(r"^(not|never)\b", pred)) or bool(re.match(r"^no\b", pred) and verb in ("have", "has", "are", "is"))
    pred_full = (verb + " " + pred) if verb not in ("are", "is", "were", "was") else pred
    S, P = _term(subj), _term(pred_full)
    if S is None or P is None:
        return None
    if kind is None:
        orig_subj = sent.strip()
        if re.match(r"^(a|an)\s", subj):
            kind = "A"          # generic indefinite ("A square is a rectangle")
        elif re.match(r"^(the|this|that)\s", subj) or re.match(r"^[A-Z][a-z]+", orig_subj):
            S = ("IND", S)
            kind = "A"
        else:
            kind = "A"          # bare plural generic ("Doctors are licensed")
    if kind == "ONLY":          # only S are P == all P are S
        if neg:
            return None
        return ("A", P, S)
    if neg:
        kind = {"A": "E", "I": "O", "E": None, "O": None}[kind]
        if kind is None:
            return None
    return (kind, S, P)


def _holds(kind, si, pi, model):
    if kind == "A":
        return not any((r >> si) & 1 and not (r >> pi) & 1 for r in model)
    if kind == "E":
        return not any((r >> si) & 1 and (r >> pi) & 1 for r in model)
    if kind == "I":
        return any((r >> si) & 1 and (r >> pi) & 1 for r in model)
    if kind == "O":
        return any((r >> si) & 1 and not (r >> pi) & 1 for r in model)
    raise ValueError(kind)


_MODELS: dict = {}


def _models(k: int):
    if k not in _MODELS:
        regs = list(range(1 << k))
        _MODELS[k] = [frozenset(r for r in regs if (mask >> r) & 1) for mask in range(1, 1 << (1 << k))]
    return _MODELS[k]


def _entailed(prem, concl, k, inds, import_=True):
    """prem: list of (kind, si, pi); concl: (kind, si, pi); k terms; inds: indices of individual terms."""
    sat = False
    for model in _models(k):
        if import_ and not all(any((r >> t) & 1 for r in model) for t in range(k)):
            continue
        if any(sum(1 for r in model if (r >> t) & 1) != 1 for t in inds):
            continue
        if not all(_holds(kd, a, b, model) for kd, a, b in prem):
            continue
        sat = True
        if not _holds(*concl, model):
            return False
    return sat


def _cat_premises(q: str):
    t = _pre_keep(q)
    head = re.split(r"\s*(?:Solely from|Based solely|Based on|From the information|Which of the following|Using only)",
                    t, maxsplit=1)[0]
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", head.strip()) if x.strip()]
    return sents


def solve_categorical_row(q: str, opts: list[str], import_: bool = True):
    sents = _cat_premises(q)
    if len(sents) != 2:
        return pd.NA, "unparsed"
    prem = [parse_cat(x) for x in sents]
    if any(p is None for p in prem):
        return pd.NA, "unparsed"
    terms: dict = {}
    inds = set()

    def key(t):
        return t[1] if isinstance(t, tuple) else t

    for kd, a, b in prem:
        for t in (a, b):
            kt = key(t)
            if kt not in terms:
                terms[kt] = len(terms)
            if isinstance(t, tuple):
                inds.add(terms[kt])
    if len(terms) > 3:
        return pd.NA, "unparsed"
    P = [(kd, terms[key(a)], terms[key(b)]) for kd, a, b in prem]
    k = len(terms)
    if not _entailed(P, ("A", 0, 0), k, inds, import_):  # A(x,x) is a tautology: False iff no model
        return pd.NA, "unparsed"  # premises unsatisfiable under this reading
    allw = set().union(*terms.keys())
    cbd = [i for i, o in enumerate(opts) if _CBD.search(o) or _POSS.search(o)]
    ent, unk, new = [], [], []
    for i, o in enumerate(opts):
        if i in cbd or not o.strip():
            continue
        c = parse_cat(o)
        if c is None:
            unk.append(i); continue
        kd, a, b = c
        ka, kb = key(a), key(b)
        known = [kx in terms for kx in (ka, kb)]
        if not all(known):
            # a term that shares words with a premise term is probably a paraphrase -> cannot decide
            if any((not kn) and (kx & allw) for kn, kx in zip(known, (ka, kb))):
                unk.append(i)
            else:
                new.append(i)
            continue  # genuinely new term: nothing about it follows (unless it paraphrases a premise term)
        if isinstance(a, tuple) != (terms[ka] in inds):
            unk.append(i); continue
        if _entailed(P, (kd, terms[ka], terms[kb]), k, inds, import_):
            ent.append(i)
    if len(ent) > 1 and len({opts[i] for i in ent}) > 1:
        return pd.NA, "ill_posed:several_options_follow"
    if ent:
        return (ent[0], "ok") if len(ent) == 1 else (ent[0], "multi_match")
    if unk or new:  # a "new" term may be a world-knowledge paraphrase (quadrilateral / four sides)
        return pd.NA, "unparsed"
    if len(cbd) == 1:
        return cbd[0], "ok"
    if not cbd:
        return pd.NA, "ill_posed:no_valid_option"
    return pd.NA, "multi_match"


def solve_categorical(df: pd.DataFrame) -> pd.DataFrame:
    preds, sts = [], []
    for q, opts in zip(df.question_text, df.options):
        o = _clean_opts(opts)
        a = solve_categorical_row(q, o, True)
        b = solve_categorical_row(q, o, False)
        if a[1] == "unparsed" or b[1] == "unparsed":
            p, s_ = pd.NA, "unparsed"
        elif a[1] == b[1] and (a[0] is pd.NA or b[0] is pd.NA or a[0] == b[0]) and \
                (a[0] is pd.NA) == (b[0] is pd.NA):
            p, s_ = a
        else:  # answer depends on whether universal statements carry existential import
            p, s_ = pd.NA, "ill_posed:existential_import"
        preds.append(p); sts.append(s_)
    return _result(df.index, preds, sts)


# ============================================================================ false belief family
# domain "false-belief": unexpected-transfer stories ("X puts obj in A, Y moves it to B; where will X look?").
#   First order and the usual second order ("where will Y think X will look?") both have answer A, the
#   location the protagonist last saw.  Heuristic: the option whose phrase is mentioned first in the story.
# domain "secret": "What should you do?" -- the option that keeps the secret.
# domain "deceptive": appearance/reality -- not solved (unparsed).
_LOC_STOP = set("in on under the a an his her their its inside into onto at of to by from where he she it left "
                "there original originally other another new different same".split())


def _loc_words(t):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", _pre(t).lower())]


def _occ(sw, seq):
    L = len(seq)
    return [(i, i + L) for i in range(len(sw) - L + 1) if sw[i:i + L] == seq]


def _first_mention(q: str, opts: list[str]):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", q.strip()) if p]
    story = " ".join(parts[:-1]) if len(parts) > 1 else q
    qw = set(_loc_words(parts[-1]))
    sw = _loc_words(story)
    seqs = [[stem(w) for w in re.findall(r"[a-z0-9]+", _pre(o).lower()) if w not in _LOC_STOP] for o in opts]
    spans = [_occ(sw, s) if s else [] for s in seqs]
    res = []
    for k, sp in enumerate(spans):
        other = [x for j, s in enumerate(spans) if j != k and len(seqs[j]) > len(seqs[k]) for x in s]
        good = [a for a, b in sp if not any(a2 <= a and b <= b2 for a2, b2 in other)]
        if good:
            res.append(min(good)); continue
        allo = {w for j, s in enumerate(seqs) if j != k for w in s}
        dist = [w for w in seqs[k] if w not in allo and w not in qw]
        c = [i for i, w in enumerate(sw) if w in dist]
        res.append(min(c) if c else None)
    return res


def question_order(q: str) -> str:
    """'second_order' if the final question asks for one agent's belief about another agent, else 'first_order'."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", q.strip()) if p]
    qs = [p for p in parts if p.endswith("?")]
    lq = qs[-1] if qs else parts[-1]
    return "second_order" if _SO.search(lq) else "first_order"


_MV = (r"(?:think|thinks|expect|expects|predict|predicts|believe|believes|assume|assumes|presume|presumes|surmise|"
       r"surmises|consider|considers|anticipate|anticipates|guess|guesses|suppose|supposes|imagine|imagines|figure|"
       r"figures|say|says|tell|tells|reckon|reckons|suspect|suspects)")
_SO = re.compile(r"\b" + _MV + r"\s+(?:that\s+)?(?!to\b)(?:\w+['’]s\s+\w+|the\s+\w+|his\s+\w+|her\s+\w+|[A-Z]\w+|he|she|"
                 r"they|it)(?:\s+\w+)?\s+(?:will|would|'ll|’ll|is going to|might|should|to|first|initially|go|look|"
                 r"search|check|seek|expect)\b|according to|in \w+['’]s (?:opinion|mind|view)|\w+['’]s belief|"
                 r"\bwhere does \w+ think (?:that )?(?!he|she)\w+", re.I)

_KEEP_FW = set("""keep maintain pretend avoid reassure encourage respect divert change distract deflect act feign deny
preserve invent politely play uphold dodge withhold refrain evade don't mislead vaguely conceal refuse redirect stick
stay hide discreetly decline honor not remain protect steer abide disguise resist dismiss listen""".split())
_REV_FW = set("""tell reveal inform share spill confirm disclose spoil hint mention explain discuss congratulate admit
divulge confess ruin announce break leak slip expose reveals confide tease show accidentally excitedly blurt""".split())


def _secret_class(o: str):
    w = re.findall(r"[a-z']+", o.lower().replace("’", "'"))
    if not w:
        return None
    if w[0] in _KEEP_FW:
        return "keep"
    if w[0] in _REV_FW:
        return "rev"
    if w[0] == "give":
        if len(w) > 1 and w[1] in ("away", "in", "hints", "hint", "details"):
            return "rev"
        if re.search(r"\b(vague|non|general|noncommittal|playful|ambiguous)", o, re.I):
            return "keep"
    return None


def solve_false_belief(df: pd.DataFrame) -> pd.DataFrame:
    preds, sts = [], []
    for q, opts, dom in zip(df.question_text, df.options, df.domain):
        opts = _clean_opts(opts)
        live = [i for i, o in enumerate(opts) if o.strip()]
        if dom == "false-belief":
            if not re.search(r"\bwhere\b", q.split(".")[-1] + q[-200:], re.I):
                preds.append(pd.NA); sts.append("unparsed"); continue
            p = _first_mention(q, [opts[i] for i in live])
            if any(x is None for x in p) or len(set(p)) != len(p):
                preds.append(pd.NA); sts.append("unparsed"); continue
            preds.append(live[p.index(min(p))]); sts.append("ok")
        elif dom == "secret":
            c = [_secret_class(opts[i]) for i in live]
            if c.count("keep") == 1 and all(x == "rev" for x in c if x != "keep"):
                preds.append(live[c.index("keep")]); sts.append("ok")
            else:
                preds.append(pd.NA); sts.append("unparsed")
        else:
            preds.append(pd.NA); sts.append("unparsed")
    return _result(df.index, preds, sts)


# ---- release text fixes / ill-posed drops (keyed by false_belief question_id; FOFB/SOFB map via question text)
_FB_OPTION_TEXT = {  # (false_belief qid, bad option text) -> replacement ("" = remove option)
    ("1054_0", "Nowhere - she sees the washed jar (alternative response)"): "Nowhere - she sees the washed jar",
    ("1220_0", "Nowhere, she will see that the jar is empty (misleading option)"): "Nowhere, she will see that the jar is empty",
    ("2550_0", 'Anywhere else (incorrect option, as "anywhere else" he looks will not show him the status of his game)'): "Anywhere else",
    ("2595_0", "Under the magazine (incorrect option, since Emma did not see the action)"): "Under the magazine",
    ("1567_0", "In the garage (ambiguous, requires clarification)"): "In the garage",
    ("1567_0", 'te: For clarity and to follow the format, option A should specify the original location, e.g., "On the garage floor.")'): "",
}
_FB_KEEP_FIRST = {"2722_0": 2}  # 7 options, 5 of them leaked generation text ("mple 7:", next example's story)
_FB_DROP = {
    "262_0": "ill_posed:asks which book a child thinks is not real; child has no false belief, keyed answer 'Any other book'",
    "2652_0": "ill_posed:second-order belief of a roommate who never saw the original placement",
    "2653_0": "ill_posed:story contradicts itself (controller stashed in drawer, found on couch) and friend never saw placement",
    "5095_0": "ill_posed:both options keep the secret",
}

REPAIRS = {
    "garbage_option_removed": "option text that is leaked generation text (e.g. 'rrect option: A)', 'B', 'mple 7:') removed",
    "option_text_fixed": "answer-leaking annotation removed from an option text (e.g. '(misleading option)')",
    "tags_rederived": "tags valid/indeterminate recomputed from the (repaired) correct option",
    "type_from_text": "type derived from question text (first_order / second_order false belief)",
    "dedupe_prefer_verified": "duplicate question texts: the solver-verified variant was kept instead of the first",
}


def _add_rep(df, mask, rid):
    df.loc[mask, "repair"] = df.loc[mask, "repair"].map(lambda r: list(r) + [rid])


def _fb_fix_rows(df: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Apply the false_belief text fixes/drops to rows whose false_belief qid is `key`."""
    for i in df.index:
        k = key.get(i)
        if k is None:
            continue
        opts, corr = list(df.at[i, "options"]), list(df.at[i, "correct"])
        changed = garbage = False
        if k in _FB_KEEP_FIRST:
            n = _FB_KEEP_FIRST[k]
            opts, corr, garbage = opts[:n], corr[:n], True
        new_o, new_c = [], []
        for o, c in zip(opts, corr):
            rep = _FB_OPTION_TEXT.get((k, o))
            if rep is None:
                new_o.append(o); new_c.append(c); continue
            if rep == "":
                garbage = True; continue
            new_o.append(rep); new_c.append(c); changed = True
        if changed or garbage:
            df.at[i, "options"] = new_o
            df.at[i, "correct"] = new_c
            df.at[i, "repair"] = list(df.at[i, "repair"]) + (["option_text_fixed"] if changed else []) + \
                (["garbage_option_removed"] if garbage else [])
        if k in _FB_DROP:
            df.at[i, "drop"] = _FB_DROP[k]
    return df


def _fb_type(df: pd.DataFrame) -> pd.DataFrame:
    m = df.domain == "false-belief"
    df.loc[m, "type"] = df.loc[m, "question_text"].map(question_order)
    _add_rep(df, m, "type_from_text")
    return df


def fix_false_belief(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _fb_fix_rows(df, df.rel_qid)
    return _fb_type(df)


def _rebuild_from_false_belief(df: pd.DataFrame) -> pd.DataFrame:
    """FOFB/SOFB: options/answers were misassigned in the release; every question text occurs exactly once in
    false_belief, whose options/answers are intact (verified against the STEER-benchmark pre-release copies)."""
    from common import SOURCE, normalize
    fb = normalize("false_belief")
    by_text = fb.drop_duplicates("question_text").set_index("question_text")
    df = df.copy()
    hit = df.question_text.isin(by_text.index)
    if not hit.all():
        raise RuntimeError(f"{(~hit).sum()} questions not found in false_belief")
    src = by_text.loc[df.question_text]
    df["options"] = [list(x) for x in src.options]
    df["correct"] = [list(x) for x in src.correct]
    df["option_ids"] = [list(x) for x in src.option_ids]
    df["n_correct"] = src.n_correct.to_numpy()
    df["n_answered"] = src.n_answered.to_numpy()
    df["source"] = SOURCE + "+false_belief"
    _add_rep(df, df.index == df.index, "rebuilt_from_false_belief")
    df = _fb_fix_rows(df, pd.Series(src.rel_qid.to_numpy(), index=df.index))
    return _fb_type(df)


# ============================================================================ logic element fixes
_GARBAGE_OPT = re.compile(r"^\s*(?:[A-E]|rrect option:?\s*[A-E]\)?|correct option:?\s*[A-E]\)?)\s*$")


def _remove_garbage_options(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in df.index:
        opts, corr = list(df.at[i, "options"]), list(df.at[i, "correct"])
        keep = [j for j, o in enumerate(opts) if not _GARBAGE_OPT.match(o)]
        if len(keep) != len(opts):
            df.at[i, "options"] = [opts[j] for j in keep]
            df.at[i, "correct"] = [corr[j] for j in keep]
            df.at[i, "repair"] = list(df.at[i, "repair"]) + ["garbage_option_removed"]
    return df


def _prefer_verified_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """POSTFIX: the release repeats question texts with different option sets. Before the pipeline's text dedupe
    (which keeps the first occurrence), keep a solver-verified variant ("ok") where one exists."""
    live = df[df["drop"].isna()]
    for t, g in live.groupby("question_text", sort=False):
        if len(g) < 2 or not (g.status == "ok").any() or g.status.iloc[0] == "ok":
            continue
        kb = g.loc[g.status == "ok", "base_id"].iloc[0]
        others = g.index[g.base_id != kb]
        df.loc[others, "drop"] = "dedup_of:" + kb
        ki = g.index[g.base_id == kb]
        _add_rep(df, df.index.isin(ki), "dedupe_text")
        _add_rep(df, df.index.isin(ki), "dedupe_prefer_verified")
    return df


def _logic_postfix(df: pd.DataFrame) -> pd.DataFrame:
    df = _prefer_verified_duplicate(df)
    if "valid" in set(x for t in df.tags for x in t) or "indeterminate" in set(x for t in df.tags for x in t):
        df = _retag_syllogism(df)
    return df


def _retag_syllogism(df: pd.DataFrame) -> pd.DataFrame:
    """POSTFIX: tags valid/indeterminate must agree with the final answer ('Cannot be determined' <-> indeterminate)."""
    ok = df.correct_text.notna()
    new = np.where(df.correct_text.fillna("").str.contains(_CBD), "indeterminate", "valid")
    old = df.tags.map(lambda t: "indeterminate" if "indeterminate" in t else ("valid" if "valid" in t else None))
    ch = ok & (old != new)
    df.loc[ch, "tags"] = pd.Series(new, index=df.index)[ch].map(lambda t: [t])
    _add_rep(df, ch, "tags_rederived")
    return df


# ============================================================================ level_k
def solve_level_k(df: pd.DataFrame) -> pd.DataFrame:
    """p-beauty contest 'closest to a/b of the average', integers in [lo, hi]: with a/b < 1 the unique Nash
    equilibrium (and the limit of level-k reasoning) is everybody choosing lo."""
    preds, sts = [], []
    for q, opts in zip(df.question_text, df.options):
        opts = _clean_opts(opts)
        r = re.search(r"between (\d+) and (\d+)", q)
        f = re.search(r"closest to (\d+)/(\d+) of the average", q)
        if not r or not f:
            preds.append(pd.NA); sts.append("unparsed"); continue
        lo, hi, a, b = map(int, (*r.groups(), *f.groups()))
        if not (a < b and lo < hi):
            preds.append(pd.NA); sts.append("ill_posed:fraction_not_below_one"); continue
        m = [i for i, o in enumerate(opts) if o.strip() == str(lo)]
        if len(m) == 1:
            preds.append(m[0]); sts.append("ok")
        else:
            preds.append(pd.NA); sts.append("no_match" if not m else "multi_match")
    return _result(df.index, preds, sts)


# ============================================================================ registry
SOLVERS = {
    "categorical_syllogism": solve_categorical,
    "conditional_syllogism": solve_conditional,
    "equiv_contrapositive": solve_contrapositive,
    "false_belief": solve_false_belief,
    "first_order_false_belief": solve_false_belief,
    "second_order_false_belief": solve_false_belief,
    "level_k": solve_level_k,
}

_LOGIC_POLICY = {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"], "shuffle": None}
_FB_POLICY = {"grading": "answer", "authoritative": False, "drop_status": ["ill_posed"], "shuffle": None}
POLICY = {
    "categorical_syllogism": dict(_LOGIC_POLICY),
    "conditional_syllogism": dict(_LOGIC_POLICY),
    "equiv_contrapositive": dict(_LOGIC_POLICY),
    "false_belief": dict(_FB_POLICY),
    "first_order_false_belief": dict(_FB_POLICY),
    "second_order_false_belief": dict(_FB_POLICY),
    "level_k": {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"], "shuffle": None},
}

FIXES = {
    "false_belief": fix_false_belief,
    "first_order_false_belief": _rebuild_from_false_belief,
    "second_order_false_belief": _rebuild_from_false_belief,
    "categorical_syllogism": _remove_garbage_options,
    "equiv_contrapositive": _remove_garbage_options,
}

POSTFIX = {
    "categorical_syllogism": _logic_postfix,
    "conditional_syllogism": _logic_postfix,
    "equiv_contrapositive": _logic_postfix,
}
