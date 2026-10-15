"""welfare_theorem_1 / welfare_theorem_2: yes/no questions, classified from the final question sentence.

First welfare theorem: a competitive equilibrium allocation is Pareto efficient, so "can further mutually
beneficial / improving trades be found after the market clears?" -> "no".
Second welfare theorem: a Pareto-efficient allocation can be supported as a competitive equilibrium
(with convex preferences, *after suitable lump-sum redistribution of endowments*) -> "yes".
The note flags WT2 questions that insist on reaching the allocation "from the initial holdings" without
mentioning transfers/redistribution: strictly, the theorem does not guarantee that (see REPORT).
"""
import re

from .common import Result, register, text_result

_Q = re.compile(r"[^.?!]*\?\s*$", re.S)


def _last_question(q: str) -> str:
    m = _Q.search(q.strip())
    return (m.group(0) if m else q).lower()


@register("welfare_theorem_1")
def welfare_theorem_1(q, options, qtype):
    last = _last_question(q)
    if re.search(r"\b(is|was|will) (this|the|that)\b[^?]*pareto[- ](efficient|optimal)", last):
        return text_result("yes", options, note="asks whether the CE allocation is Pareto efficient")
    if re.search(r"trade|exchange|swap|better off|improve|advantageous|beneficial|better outcome|reallocat", last):
        return text_result("no", options)
    return Result("unparsed", note="final question not recognised")


@register("welfare_theorem_2")
def welfare_theorem_2(q, options, qtype):
    last = _last_question(q)
    # me-fillers variant: "Without any redistribution ..., is it guaranteed ...?" -> no
    if re.search(r"without (any )?(redistribut|transfer)", last) and re.search(r"guarantee", last):
        return text_result("no", options, note="no transfers: the second theorem gives no guarantee")
    if re.search(r"price", last):
        low = q.lower()
        note = ""
        if re.search(r"initial|original|starting|given their", low) and not re.search(
                r"transfer|redistribut|reallocat(e|ion) of (the )?(initial )?endowment|lump", low):
            note = "asks for prices supporting the allocation from the initial endowment without transfers"
        return text_result("yes", options, note=note)
    return Result("unparsed", note="final question not recognised")
