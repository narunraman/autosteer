"""Scoring of consistency-graded STEER items (rows with grading == "consistency", correct_index = null).

Rule strings follow the solvers' convention (tests/solvers/steer/solvers/biases.py):

    consistency_rule = "<rule_id>:<allowed>"

where <allowed> lists the consistent choices as option indices into the *released* (already shuffled)
`options` of the rows of one consistency_group, ordered by part.

Indifference (October 2026). ellsberg_seu, same_brand and mirror_trade groups generated from October 2026 on
offer an "indifferent" option in each part, and their rule has a third index pair, the indices of those options:
"<rule_id>:<pair>,<pair>,<i0><i1>". Indifferent in both parts is consistent; indifferent in exactly one part is
"indifference_mismatch" (an agent indifferent in one part is, under the rule's model, indifferent in the other).
Without the option the patterns these rules call biases can be ties: at the belief p = 1/3 an SEU agent is indifferent
in both Ellsberg parts, an agent indifferent between two items of equal value may decline both swaps, and a brand
tie may be broken by the decoy. A strict choice now states a strict preference. Rules with two pairs (no
indifference option) are scored as before.

Rule ids defined here
---------------------
common_ratio_same_side   certainty_effect (Allais / common-ratio pairs; Kahneman & Tversky).
    Group = two rows of one base_id:
        part 0:  A = X for sure              vs  B = Y with probability p, else nothing   (Y > X > 0)
        part 1:  C = X with probability k    vs  D = Y with probability k*p, else nothing (0 < k < 1)
    Under expected utility (u increasing, u(0) = 0), A > B  <=>  u(X) > p u(Y)  <=>  C > D, so the rule is
    "pick the option in part 1 on the same side as your part-0 choice".
    Encoding: "common_ratio_same_side:<a><c>,<b><d>", i.e. exactly two index pairs, the FIRST is the
    sure-side pair (index of A in part 0, index of C in part 1), the second the risky-side pair (B, D).
    Example: "common_ratio_same_side:10,01" -> A is option 1 of part 0 and C is option 0 of part 1.
    Outcomes of a pair of choices (c0, c1):
        (A, C) or (B, D)  -> "consistent"
        (A, D)            -> "certainty_effect"   (the classic Allais violation)
        (B, C)            -> "reverse"            (the reverse violation)

reflection_same_side     reflection_effect (gain / loss mirror pairs; Kahneman & Tversky 1984, STEER 2.5.c).
    Group = two rows of one base_id, same story, same X < Y and p, with EXACTLY equal expected values p*Y = X:
        part 0 (gains):   A = gain X for sure    vs  B = gain Y with probability p, else nothing
        part 1 (losses):  C = lose X for sure    vs  D = lose Y with probability p, else nothing
    Under expected utility with one risk attitude over the whole range (u strictly concave throughout: sure in
    both frames by Jensen; strictly convex throughout: risky in both), the choice is on the same side in both
    frames, so (A, C) and (B, D) are consistent and a flip is the violation. (A risk-neutral agent is indifferent
    in both parts; the rule treats a flip as inconsistent anyway, i.e. it assumes a strict, frame-independent
    risk attitude. Equal EVs are essential: with p*Y != X a risk-neutral agent would itself flip.)
    Encoding as for common_ratio_same_side: "reflection_same_side:<a><c>,<b><d>", first pair = sure side.
    Outcomes: (A, C) or (B, D) -> "consistent"; (A, D) -> "reflection" (risk-averse for gains, risk-seeking for
    losses, the classic pattern); (B, C) -> "reverse_reflection".

ellsberg_seu             ambiguity_aversion (Ellsberg's three-colour urn; STEER 2.5.d).
    Group = two rows of one base_id. The urn holds n balls of a known colour K and 2n balls of two other colours
    U1, U2 in unknown proportions; every bet pays the same reward.
        part 0:  bet on K (probability 1/3)              vs  bet on U1 (subjective probability p)
        part 1:  bet on U1 or U2 (probability 2/3)       vs  bet on K or U2 (subjective probability 1 - p)
    Under subjective expected utility with one belief p: K over U1 iff p < 1/3 iff K-or-U2 over U1-or-U2. So the
    consistent pairs are (K, K or U2) and (U1, U1 or U2): the known-probability bet in exactly one of the two
    parts. (At p = 1/3 the agent is indifferent in both parts; the rule lists the pairs a strict preference
    allows.) Encoding: "ellsberg_seu:<a><c>,<b><d>" with (a, c) = (index of K in part 0, index of K-or-U2 in part 1)
    and (b, d) = (index of U1, index of U1-or-U2). Outcomes: a listed pair -> "consistent"; the known-probability
    bet in both parts (K, U1-or-U2) -> "ambiguity_aversion" (Ellsberg's pattern); in neither (U1, K-or-U2) ->
    "ambiguity_seeking".

same_brand               independence (the decoy effect; STEER 2.1.b).
    Group = two rows of one base_id. Part 0 offers brands A and B (price and quality); part 1 offers A, B and a
    decoy C that is worse than the target (one of A, B) on both price and quality but not worse than the other
    brand on both. Choices are consistent iff the same brand is chosen in both parts (and the decoy never is).
    Encoding: "same_brand:<t0><t1>,<o0><o1>": (released index of the target in part 0, in part 1), then the same
    for the other brand. Outcomes: a listed pair -> "consistent"; other brand then target -> "decoy_effect"
    (the attraction effect); target then other brand -> "reverse"; the decoy in part 1 -> "decoy_chosen".

mirror_trade             endowment_effect (STEER 2.2.b).
    Group = two rows of one base_id, the same story told twice with the two items swapped, each with options Yes / No:
        part 0:  you own X; you are offered Y for it (of the same stated value). Do you accept?
        part 1:  you own Y; you are offered X for it. Do you accept?
    With a strict preference between X and Y that does not depend on which one is owned, exactly one of the two
    offers is accepted: (Yes, No) if Y is preferred, (No, Yes) if X is. Encoding: "mirror_trade:<y0><n1>,<n0><y1>",
    i.e. the pair (index of Yes in part 0, index of No in part 1), then (No in part 0, Yes in part 1). Outcomes: a
    listed pair -> "consistent"; No in both parts (keep whichever item is owned) -> "endowment_effect"; Yes in both
    -> "reverse_endowment". (Indifference between X and Y would allow any answers; the rule, like the others here,
    assumes a strict preference.)

pairwise_complete        completeness, completeness_risk (STEER 2.1.a, 2.3.a; design of October 2026).
    Group = one row per pair of a set of n fully described alternatives (n = 4 or 5, every pair asked once). Each part
    offers four options: prefer the first alternative of the pair, prefer the second, indifferent, and "cannot decide
    / need more information". Unlike the rules above, the tokens are not allowed tuples: there is one token per part,
    in part order, of six digits "abpqic": a < b are the alternatives the part compares (0-based, in the order they
    are described), then the released option indices of "prefer a", "prefer b", "indifferent" and "cannot decide".
    Completeness (the score): the choices are "consistent" (complete) iff no part is answered "cannot decide";
    otherwise "incomplete". Transitivity (secondary, score_transitivity): the answered pairs (strict preference or
    indifference; "cannot decide" parts left out) are "transitive" iff on every triple whose three pairs are all
    answered the weak preference is transitive (x >= y and y >= z imply x >= z), else "intransitive". A complete and
    transitive set of answers is a weak order over the alternatives.
"""
from __future__ import annotations

from typing import Optional

COMMON_RATIO = "common_ratio_same_side"
REFLECTION = "reflection_same_side"
CONSISTENT, CERTAINTY_EFFECT, REVERSE, INVALID = "consistent", "certainty_effect", "reverse", "invalid"
REFLECTION_EFFECT, REVERSE_REFLECTION = "reflection", "reverse_reflection"

# rule id -> (label for sure-then-risky, label for risky-then-sure)
SAME_SIDE_RULES = {
    COMMON_RATIO: (CERTAINTY_EFFECT, REVERSE),
    REFLECTION: (REFLECTION_EFFECT, REVERSE_REFLECTION),
}


ELLSBERG = "ellsberg_seu"
SAME_BRAND = "same_brand"
AMBIGUITY_AVERSION, AMBIGUITY_SEEKING = "ambiguity_aversion", "ambiguity_seeking"
DECOY_EFFECT, DECOY_CHOSEN = "decoy_effect", "decoy_chosen"
MIRROR_TRADE = "mirror_trade"
ENDOWMENT_EFFECT, REVERSE_ENDOWMENT = "endowment_effect", "reverse_endowment"


def parse_rule(rule: str) -> tuple[str, list[tuple[int, ...]]]:
    """'rule_id:01,10' -> ('rule_id', [(0, 1), (1, 0)])."""
    rid, _, allowed = rule.partition(":")
    if not rid or not allowed:
        raise ValueError(f"malformed consistency rule {rule!r}")
    pairs = [tuple(int(ch) for ch in tok) for tok in allowed.split(",")]
    return rid, pairs


def same_side_rule(rule_id: str, sure0: int, sure1: int) -> str:
    """Rule string for a two-part, two-option group whose sure-side options sit at released indices sure0 / sure1."""
    if rule_id not in SAME_SIDE_RULES:
        raise ValueError(f"unknown same-side rule {rule_id!r}")
    if sure0 not in (0, 1) or sure1 not in (0, 1):
        raise ValueError("same-side parts have exactly two options")
    return f"{rule_id}:{sure0}{sure1},{1 - sure0}{1 - sure1}"


def common_ratio_rule(sure0: int, sure1: int) -> str:
    return same_side_rule(COMMON_RATIO, sure0, sure1)


def reflection_rule(sure0: int, sure1: int) -> str:
    return same_side_rule(REFLECTION, sure0, sure1)


def sure_side(rule: str, rule_id: Optional[str] = None) -> tuple[int, int]:
    """(index of the sure-side option in part 0, in part 1) from a same-side rule (optionally of a given id)."""
    rid, pairs = parse_rule(rule)
    if rid not in SAME_SIDE_RULES or (rule_id is not None and rid != rule_id):
        raise ValueError(f"not a {rule_id or 'same-side'} rule: {rule!r}")
    if len(pairs) != 2 or any(len(p) != 2 for p in pairs):
        raise ValueError(f"{rid} needs exactly two index pairs: {rule!r}")
    (a, c), (b, d) = pairs
    if {a, b} != {0, 1} or {c, d} != {0, 1}:
        raise ValueError(f"{rid} pairs must be complementary: {rule!r}")
    return a, c


def score_same_side(rule: str, choice0: Optional[int], choice1: Optional[int], rule_id: Optional[str] = None) -> str:
    """Classify choices (released option indices) for part 0 and part 1 of a same-side group: "consistent", the
    rule's sure-then-risky label, its risky-then-sure label, or "invalid" (missing / not 0 or 1)."""
    a, c = sure_side(rule, rule_id)
    if choice0 not in (0, 1) or choice1 not in (0, 1) or isinstance(choice0, bool) or isinstance(choice1, bool):
        return INVALID
    sure0, sure1 = choice0 == a, choice1 == c
    if sure0 == sure1:
        return CONSISTENT
    sure_then_risky, risky_then_sure = SAME_SIDE_RULES[parse_rule(rule)[0]]
    return sure_then_risky if sure0 else risky_then_sure


def score_common_ratio(rule: str, choice0: Optional[int], choice1: Optional[int]) -> str:
    """certainty_effect: "consistent", "certainty_effect" (sure then risky), "reverse", or "invalid"."""
    return score_same_side(rule, choice0, choice1, COMMON_RATIO)


def score_reflection(rule: str, choice0: Optional[int], choice1: Optional[int]) -> str:
    """reflection_effect: "consistent", "reflection" (sure gain, risky loss), "reverse_reflection", or "invalid"."""
    return score_same_side(rule, choice0, choice1, REFLECTION)


INDIFFERENCE_MISMATCH = "indifference_mismatch"


def _choice_ok(c, n: int) -> bool:
    return c is not None and not isinstance(c, bool) and c in range(n)


def _pairs_with_indifference(rule: str, rule_id: str, n0: int, n1: int):
    """Parse a two-part rule: two strict index pairs, optionally a third pair = the indifference options (added in
    October 2026). Returns (pair, pair, indifference pair or None, option count part 0, part 1). n0 / n1 are the
    option counts without an indifference option."""
    rid, pairs = parse_rule(rule)
    if rid != rule_id or len(pairs) not in (2, 3) or any(len(p) != 2 for p in pairs):
        raise ValueError(f"not a {rule_id} rule: {rule!r}")
    ind = pairs[2] if len(pairs) == 3 else None
    n0, n1 = (n0 + 1, n1 + 1) if ind else (n0, n1)
    col0, col1 = [p[0] for p in pairs], [p[1] for p in pairs]
    if len(set(col0)) != len(col0) or len(set(col1)) != len(col1) or max(col0) >= n0 or max(col1) >= n1:
        raise ValueError(f"{rule_id} pairs must use distinct option indices: {rule!r}")
    return pairs[0], pairs[1], ind, n0, n1


def _with_indifference(rule: str, ind: Optional[tuple[int, int]]) -> str:
    return rule if ind is None else f"{rule},{ind[0]}{ind[1]}"


def _score_indifference(ind, choice0, choice1) -> Optional[str]:
    """CONSISTENT if indifferent in both parts, INDIFFERENCE_MISMATCH if in exactly one, None if in neither."""
    if ind is None:
        return None
    i0, i1 = choice0 == ind[0], choice1 == ind[1]
    if i0 and i1:
        return CONSISTENT
    return INDIFFERENCE_MISMATCH if (i0 or i1) else None


def ellsberg_rule(known0: int, known1: int, other0: Optional[int] = None, other1: Optional[int] = None,
                  indifferent: Optional[tuple[int, int]] = None) -> str:
    """Rule for an Ellsberg pair whose known-probability bets sit at released indices known0 (the known colour, part
    0) and known1 (the two unknown colours together, part 1); other0 / other1 are the other bets (default: the other
    of two options), `indifferent` the indices of the "indifferent" options, if any."""
    other0 = 1 - known0 if other0 is None else other0
    other1 = 1 - known1 if other1 is None else other1
    n = 2 if indifferent is None else 3
    if len({known0, other0}) != 2 or len({known1, other1}) != 2 or not {known0, other0, known1, other1} <= set(range(n)):
        raise ValueError("Ellsberg parts have two bets (and possibly an indifference option)")
    return _with_indifference(f"{ELLSBERG}:{known0}{other1},{other0}{known1}", indifferent)


def ellsberg_known(rule: str) -> tuple[int, int]:
    """(index of the known-probability bet in part 0, in part 1) from an ellsberg_seu rule."""
    (a, _), (_, d), _, _, _ = _pairs_with_indifference(rule, ELLSBERG, 2, 2)
    return a, d


def score_ellsberg(rule: str, choice0: Optional[int], choice1: Optional[int]) -> str:
    """"consistent", "ambiguity_aversion" (known-probability bet in both parts), "ambiguity_seeking",
    "indifference_mismatch" (indifferent in exactly one part) or "invalid"."""
    (k0, _), (_, k1), ind, n0, n1 = _pairs_with_indifference(rule, ELLSBERG, 2, 2)
    if not _choice_ok(choice0, n0) or not _choice_ok(choice1, n1):
        return INVALID
    by_ind = _score_indifference(ind, choice0, choice1)
    if by_ind:
        return by_ind
    known0, known1 = choice0 == k0, choice1 == k1
    if known0 != known1:
        return CONSISTENT
    return AMBIGUITY_AVERSION if known0 else AMBIGUITY_SEEKING


def same_brand_rule(target0: int, target1: int, other0: int, other1: int,
                    indifferent: Optional[tuple[int, int]] = None) -> str:
    """Rule for a decoy pair: released indices of the target brand and of the other brand in part 0 and part 1, and
    of the "no preference" options if any."""
    n0, n1 = (2, 3) if indifferent is None else (3, 4)
    used0 = [target0, other0] + ([indifferent[0]] if indifferent else [])
    used1 = [target1, other1] + ([indifferent[1]] if indifferent else [])
    if len(set(used0)) != len(used0) or set(used0) != set(range(n0)) or len(set(used1)) != len(used1) \
            or not set(used1) <= set(range(n1)):
        raise ValueError("decoy parts have two and three options (plus an indifference option); the brands are distinct")
    return _with_indifference(f"{SAME_BRAND}:{target0}{target1},{other0}{other1}", indifferent)


def score_same_brand(rule: str, choice0: Optional[int], choice1: Optional[int]) -> str:
    """"consistent", "decoy_effect" (other brand, then the target), "reverse", "decoy_chosen",
    "indifference_mismatch" or "invalid"."""
    (t0, t1), (o0, o1), ind, n0, n1 = _pairs_with_indifference(rule, SAME_BRAND, 2, 3)
    decoy = (set(range(n1)) - {t1, o1} - ({ind[1]} if ind else set())).pop()
    if not _choice_ok(choice0, n0) or not _choice_ok(choice1, n1):
        return INVALID
    if choice1 == decoy:
        return DECOY_CHOSEN
    by_ind = _score_indifference(ind, choice0, choice1)
    if by_ind:
        return by_ind
    if (choice0, choice1) in ((t0, t1), (o0, o1)):
        return CONSISTENT
    return DECOY_EFFECT if choice1 == t1 else REVERSE


def mirror_trade_rule(yes0: int, yes1: int, no0: Optional[int] = None, no1: Optional[int] = None,
                      indifferent: Optional[tuple[int, int]] = None) -> str:
    """Rule for an endowment pair whose "Yes" options sit at released indices yes0 (part 0) and yes1 (part 1); no0 /
    no1 are the "No" options (default: the other of two), `indifferent` the "indifferent" options, if any."""
    no0 = 1 - yes0 if no0 is None else no0
    no1 = 1 - yes1 if no1 is None else no1
    n = 2 if indifferent is None else 3
    if len({yes0, no0}) != 2 or len({yes1, no1}) != 2 or not {yes0, no0, yes1, no1} <= set(range(n)):
        raise ValueError("mirror-trade parts have Yes and No (and possibly an indifference option)")
    return _with_indifference(f"{MIRROR_TRADE}:{yes0}{no1},{no0}{yes1}", indifferent)


def mirror_trade_yes(rule: str) -> tuple[int, int]:
    """(index of Yes in part 0, in part 1) from a mirror_trade rule."""
    (y0, _), (_, y1), _, _, _ = _pairs_with_indifference(rule, MIRROR_TRADE, 2, 2)
    return y0, y1


def score_mirror_trade(rule: str, choice0: Optional[int], choice1: Optional[int]) -> str:
    """"consistent", "endowment_effect" (No in both parts), "reverse_endowment" (Yes in both),
    "indifference_mismatch" or "invalid"."""
    (y0, _), (_, y1), ind, n0, n1 = _pairs_with_indifference(rule, MIRROR_TRADE, 2, 2)
    if not _choice_ok(choice0, n0) or not _choice_ok(choice1, n1):
        return INVALID
    by_ind = _score_indifference(ind, choice0, choice1)
    if by_ind:
        return by_ind
    yes0, yes1 = choice0 == y0, choice1 == y1
    if yes0 != yes1:
        return CONSISTENT
    return REVERSE_ENDOWMENT if yes0 else ENDOWMENT_EFFECT


PAIRWISE_COMPLETE = "pairwise_complete"
INCOMPLETE = "incomplete"
TRANSITIVE, INTRANSITIVE = "transitive", "intransitive"


def pairwise_complete_rule(parts: list[tuple[int, int, int, int, int, int]]) -> str:
    """Rule for a completeness group: per part (a, b, index of "prefer a", "prefer b", "indifferent", "cannot
    decide"), released indices. Every pair of the alternatives 0..n-1 must be asked exactly once."""
    pairs = [(a, b) for a, b, *_ in parts]
    n = max(b for _, b in pairs) + 1 if pairs else 0
    if n > 10 or sorted(pairs) != [(a, b) for a in range(n) for b in range(a + 1, n)]:
        raise ValueError(f"{PAIRWISE_COMPLETE}: every pair of the alternatives must be asked once: {pairs}")
    for a, b, *idx in parts:
        if sorted(idx) != [0, 1, 2, 3]:
            raise ValueError(f"{PAIRWISE_COMPLETE}: each part has four distinct options: {idx}")
    return f"{PAIRWISE_COMPLETE}:" + ",".join("".join(str(d) for d in p) for p in parts)


def parse_pairwise_complete(rule: str) -> list[tuple[int, int, int, int, int, int]]:
    rid, _, allowed = rule.partition(":")
    if rid != PAIRWISE_COMPLETE:
        raise ValueError(f"not a {PAIRWISE_COMPLETE} rule: {rule!r}")
    parts = [tuple(int(ch) for ch in tok) for tok in allowed.split(",")]
    if any(len(p) != 6 for p in parts):
        raise ValueError(f"{PAIRWISE_COMPLETE} tokens have six digits: {rule!r}")
    pairwise_complete_rule(parts)  # validates
    return parts


def _pairwise_answers(rule: str, choices: list[Optional[int]]):
    """{(a, b): '>' | '<' | '=' | '?'} per part, or None if a choice is missing or out of range."""
    parts = parse_pairwise_complete(rule)
    if len(choices) != len(parts) or not all(_choice_ok(c, 4) for c in choices):
        return None
    out = {}
    for (a, b, pa, pb, ind, cant), c in zip(parts, choices):
        out[(a, b)] = {pa: '>', pb: '<', ind: '=', cant: '?'}[c]
    return out


def score_completeness(rule: str, choices: list[Optional[int]]) -> str:
    """"consistent" (no pair answered "cannot decide"), "incomplete" or "invalid"."""
    ans = _pairwise_answers(rule, choices)
    if ans is None:
        return INVALID
    return INCOMPLETE if '?' in ans.values() else CONSISTENT


def score_transitivity(rule: str, choices: list[Optional[int]]) -> str:
    """Secondary rule of a pairwise_complete group: "transitive" if the weak preference given by the answered pairs
    is transitive on every triple whose three pairs are answered, "intransitive" if not, "invalid" otherwise."""
    ans = _pairwise_answers(rule, choices)
    if ans is None:
        return INVALID
    geq = set()  # (x, y): x weakly preferred to y
    for (a, b), v in ans.items():
        if v in ('>', '='):
            geq.add((a, b))
        if v in ('<', '='):
            geq.add((b, a))
    answered = {frozenset(p) for p, v in ans.items() if v != '?'}
    n = max(b for _, b in ans) + 1
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if len({x, y, z}) < 3 or not {frozenset((x, y)), frozenset((y, z)), frozenset((x, z))} <= answered:
                    continue
                if (x, y) in geq and (y, z) in geq and (x, z) not in geq:
                    return INTRANSITIVE
    return TRANSITIVE


def score_group(rule: str, choices: list[Optional[int]]) -> str:
    """Dispatch on the rule id. `choices` = chosen released option index per part, ordered by part."""
    if rule.partition(":")[0] == PAIRWISE_COMPLETE:
        return score_completeness(rule, choices)
    rid, _ = parse_rule(rule)
    if rid in SAME_SIDE_RULES:
        if len(choices) != 2:
            raise ValueError(f"{rid} groups have two parts, got {len(choices)} choices")
        return score_same_side(rule, *choices)
    if rid in (ELLSBERG, SAME_BRAND, MIRROR_TRADE):
        if len(choices) != 2:
            raise ValueError(f"{rid} groups have two parts, got {len(choices)} choices")
        return {ELLSBERG: score_ellsberg, SAME_BRAND: score_same_brand,
                MIRROR_TRADE: score_mirror_trade}[rid](rule, *choices)
    raise NotImplementedError(f"no scorer for consistency rule {rid!r}")
