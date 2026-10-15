"""Code templates never repeat a name where a possessive pronoun belongs ("Maria invests Maria's savings") and never
write "a" before a vowel ("a online course provider")."""
import re

from autosteer.filler_tools import code_templates as ct
from autosteer.filler_tools.utilities.story_utils import MEN, WOMEN, articles, possessive_pronouns
from autosteer.sampler.generator import ElementSampler

NAMES = MEN + WOMEN
REPEAT = re.compile(r"\b(" + "|".join(NAMES) + r"|the player)\b[^.?!]*\b\1's\b")


def test_helper():
    assert possessive_pronouns("Maria invests Maria's savings.") == "Maria invests her savings."
    assert possessive_pronouns("What should Arjun do to raise Arjun's utility?") == "What should Arjun do to raise his utility?"
    s = "if the country has high inflation, then the country's currency is depreciating"   # logic clauses kept
    assert possessive_pronouns(s) == s
    # a possessive first mention, another sentence, or another man in between: unchanged
    assert possessive_pronouns("Maria's income rose. Maria's savings fell.") == "Maria's income rose. Maria's savings fell."
    assert possessive_pronouns("If Grace's income rises, how does Grace's demand change?") == \
        "If Grace's income rises, how does her demand change?"
    assert possessive_pronouns("tails, the player pays out of the player's own pocket", {'the player': 'their'}) == \
        "tails, the player pays out of their own pocket"
    s = "Daniel pays Omar out of Daniel's pocket."
    assert possessive_pronouns(s) == s
    assert possessive_pronouns("Daniel pays Maria out of Daniel's pocket.") == "Daniel pays Maria out of his pocket."


def test_no_repeated_name_possessive_in_code_templates():
    ElementSampler()   # registers every code template
    bad = [(e, m.group(0)) for e in ct.elements() for t in ct._REGISTRY[e] for p in t.parts
           for m in [REPEAT.search(p)] if m]
    assert not bad, bad[:5]


A_VOWEL = re.compile(r"(?:\ba|(?:^|[.!?:]\s)A) (?!one\b|once\b)[aeio][a-z]")


def test_articles():
    assert articles("Amara runs a online course provider.") == "Amara runs an online course provider."
    assert articles("A animation studio has a utility and a unit.") == "An animation studio has a utility and a unit."
    s = "Then she receives an A in the course. Firm A is large, and Set A offers more."   # labels kept
    assert articles(s) == s


def test_no_a_before_vowel_in_code_templates():
    ElementSampler()
    bad = [(e, m.group(0)) for e in ct.elements() for t in ct._REGISTRY[e] for p in t.parts
           for m in [A_VOWEL.search(p)] if m]
    assert not bad, bad[:5]
