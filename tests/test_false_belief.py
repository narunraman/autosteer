"""first_order_false_belief, second_order_false_belief: keys recomputed by reading the story text.

The story is read sentence by sentence (not from the filler's event lists): who puts the object where, who moves it
and who sees the move (watching through the window, seeing it happen, watching while someone moves it), and who is
told about it (a message that is read, a phone call, being told later), including who knows that the owner was told
(the sender of a message the owner confirms, someone in front of whom the owner is phoned, someone told that the
owner has read the message). First order: the asked person looks where they last learned the object was. Second
order (redesign of October 2026): read by `read_second_order`, a separate reader of the two-move stories, which
tracks the owner's real belief and the observer's model of it (open phone calls, secret messages the observer does
not know about, a mistaken "I have told her" that the observer believes); the key is the observer's model. Checks on
1,000 seeded samples per element: the key, distinct options (the two places; second order the three places), both
kinds of key per element, both perspectives, every scenario domain, the owner never sees a move (second order), and
the same seed gives the same question. Second order also: observer_wrong questions differ from where the owner will
really look, observer_right ones from where the object is, and no single place answers more than half of them. The
audit solver (first-mentioned place) agrees on the first-order questions whose key is the original place.
"""
import collections
import re

import pandas as pd

from helpers import load_steer_solver, sample
from autosteer.sampler import ElementSampler

NAME = r"(?:[A-Z][a-z]+|you|You)"


def sentences(q):
    # split at ". " but not inside quotes
    out, cur, quoted = [], '', False
    for i, ch in enumerate(q):
        cur += ch
        if ch == '"':
            quoted = not quoted
        if ch in '.?' and not quoted and (i + 1 == len(q) or q[i + 1] == ' '):
            out.append(cur.strip())
            cur = ''
    return out


MOVE_PREP = {}


def read_story(q):
    """(owner, first place, second place, belief: person -> place, knows: (x, y) -> place x thinks y believes)."""
    sents = sentences(q)
    owner, p1 = re.search(rf'({NAME}) puts (?:his|her) .+? ((?:in|on) the [^.]+?)(?: and |\.)', q).groups()
    m = re.search(rf'({NAME}) (?:comes? in and )?moves? the [a-z ]+? (in|on)to (the [^.,]+?)(?: and |, and | while |\.)', q)
    mover, p2 = m.group(1), m.group(3)
    # the move names where the object ends up, as the option does ("onto the filing cabinet" / "On the filing
    # cabinet"; review, October 2026: "to the filing cabinet" left in or on open)
    MOVE_PREP[q] = m.group(2)
    present = set()
    m0 = re.match(rf'({NAME}) and ({NAME}) are in', q)
    if m0:
        present = {m0.group(1), m0.group(2)}
    belief = {owner: p1, **{x: p1 for x in present}}
    knows = {(x, y): p1 for x in present for y in present if x != y}
    for s in sents:
        if re.search(r'moves? the .* (?:in|on)to ', s) and '"' not in s:
            seers = {mover}
            for pat in (rf'while ({NAME}) watches', rf'({NAME}) watches as', rf'({NAME}) sees it happen'):
                seers |= set(re.findall(pat, s))
            if re.search(r'Through the window, (?:he|she) watches', s):
                seers.add(owner)
            for x in seers:
                belief[x] = p2
        m = re.search(rf'(?:({NAME})|and) sends? ({NAME}) a message', s)
        if m:
            sender, rcpt = m.group(1) or mover, m.group(2)  # "X moves ... and sends Y a message": X sends
            rcpt_reads = re.search(rf'{rcpt} reads', q) or re.search(rf'{rcpt} replies that (?:he|she) has read', q)
            if rcpt_reads:
                belief[rcpt] = p2
                if re.search(rf'{rcpt} replies that (?:he|she) has read', q):
                    knows[(sender, rcpt)] = p2
        m = re.search(rf'in front of ({NAME}), ({NAME}) phones? ({NAME}) and tells?', s)
        if m:
            witness, caller, rcpt = m.groups()
            belief[rcpt] = p2
            knows[(witness, rcpt)] = p2
            knows[(caller, rcpt)] = p2
        m = re.search(rf'({NAME}) tells? ({NAME}): "I moved the [a-z ]+ (?:in|on)to (the [^,]+), and ({NAME}) has read', s)
        if m:
            _, hearer, place, told = m.groups()
            belief[hearer] = place
            knows[(hearer, told)] = place
        m = re.search(rf'Nobody tells ({NAME})(?: or ({NAME}))?', s)
        if m:
            assert all(belief.get(x, p1) == p1 for x in m.groups() if x), q
    return owner, p1, p2, belief, knows


def test_move_names_the_option_place():
    """The move says where the object ends up ('onto the filing cabinet'), matching the option ('On the filing
    cabinet'); 'to the filing cabinet' left in or on open (review, October 2026)."""
    for el, seed in (('first_order_false_belief', 4606), ('second_order_false_belief', 4707)):
        for s in sample(el, 300, seed=seed):
            moves = re.findall(r'moves? the [a-z ]+? (?:again, )?(in|on)to (the [^,.]+?)(?: while |, and | and |\.)',
                               s.question)
            assert moves and len(moves) == (2 if el.startswith('second') else 1), s.question
            for prep, moved_to in moves:
                opt = [o for o in s.options if o.split(' ', 1)[1] == moved_to]
                assert opt and opt[0].startswith(prep.capitalize() + ' '), (s.question, s.options)


def place(option):
    return re.sub(r'^(?:In|On) ', '', option)


def test_first_order_keys():
    seen = collections.Counter()
    for s in sample('first_order_false_belief', 1000, seed=4606):
        owner, p1, p2, belief, _ = read_story(s.question)
        asked = re.search(rf'(?:Where will|Where does|When) ({NAME})', s.question.split('. ')[-1]).group(1)
        assert asked == owner
        assert sorted(place(o) for o in s.options) == sorted([p1.split(' ', 1)[1], p2])
        want = belief[owner]
        want = want.split(' ', 1)[1] if want == p1 else want
        assert place(s.answer) == want, (s.question, s.options)
        seen[s.type, want == p2] += 1
        seen[s.metadata['tags'].split(';')[0]] += 1
        seen[s.metadata['domain']] += 1
    assert seen[('false_belief', False)] > 300 and seen[('true_belief', True)] > 300, seen
    assert seen[('false_belief', True)] == seen[('true_belief', False)] == 0
    assert seen['third_person'] > 300 and seen['second_person'] > 300
    assert {d for d in seen if isinstance(d, str)} >= {'home', 'office', 'school', 'library', 'workplace', 'travel',
                                                         'garden', 'restaurant'}


def read_second_order(q):
    """(owner, observer, places in order, where the owner thinks it is, where the observer thinks the owner thinks it
    is, where it is) from a second-order story."""
    m = re.match(rf'({NAME}) and ({NAME}) are in .+?\. \1 puts (?:his|her) [a-z ]+? ((?:in|on) [^.]+?) and leaves', q)
    owner, observer, first = m.group(1), m.group(2), m.group(3).split(' ', 1)[1]
    assert re.search(rf'{observer} stays in', q)
    places = [first]
    owner_bel, model = first, first
    for s in sentences(q):
        mv = re.search(r'moves? the [a-z ]+? (?:again, )?(?:in|on)to (the [^,.]+?)(?: while |, and |\.)', s)
        if mv:
            places.append(mv.group(1))
            assert re.search(rf'while {observer} watches|{observer} sees this too', s), s
            continue
        here = places[-1]
        if re.search(rf'in front of {observer}, {NAME} phones? {owner} and tells? (?:him|her) that the [a-z ]+ is now (?:in|on) (the [^.]+)', s):
            owner_bel = model = here
        elif re.search(rf'sends? {owner} a message about (?:this move|the second move), which {owner} reads at once; {observer} does not know', s):
            owner_bel = here
        elif re.search(rf'"I have just told {owner} where the [a-z ]+ is now\."', s):
            assert re.search(rf'{observer} believes', q) and re.search(rf'{owner} never learns about the second move', q)
            model = here
        elif s.startswith(f'Nobody tells {owner}'):
            pass
    return owner, observer, places, owner_bel, model, places[-1]


def test_second_order_keys():
    seen = collections.Counter()
    for s in sample('second_order_false_belief', 1000, seed=4707):
        owner, observer, places, owner_bel, model, real = read_second_order(s.question)
        last = s.question.split('. ')[-1]
        m = re.search(rf'Where does ({NAME}) think ({NAME}) will look', last) or \
            re.search(rf'When ({NAME}) returns, where does ({NAME}) expect', last)
        thinker, about = (m.group(1), m.group(2)) if m.re.pattern.startswith('Where') else (m.group(2), m.group(1))
        assert about == owner and thinker == observer
        assert re.search(r'While (?:he|she) is away', s.question)  # the owner is away for both moves
        assert len(places) == 3 and len(set(places)) == 3
        assert sorted(place(o) for o in s.options) == sorted(places)
        assert place(s.answer) == model, (s.question, s.options)
        if s.type == 'observer_wrong':
            assert model != owner_bel
        else:
            assert s.type == 'observer_right' and model == owner_bel and model != real
        assert model != real or model != owner_bel        # never answered by where it is and where the owner looks
        seen[s.type] += 1
        seen['key', places.index(model)] += 1
        seen['first', model == places[0]] += 1
        seen['real', model == real] += 1
        seen['owner', model == owner_bel] += 1
        seen[s.metadata['tags'].split(';')[0]] += 1
    assert seen['observer_wrong'] > 550 and seen['observer_right'] > 200, seen
    assert all(seen['key', i] > 200 for i in range(3)), seen
    assert seen['first', True] < 500 and seen['real', True] < 350 and seen['owner', True] < 350, seen
    assert seen['third_person'] > 300 and seen['second_person'] > 300


def test_audit_solver_on_false_belief_items():
    """The audit solver (logic_tom.solve_false_belief) answers with the place mentioned first, which is right
    exactly when the asked belief is the original place; it must agree on every such question it answers."""
    logic = load_steer_solver('logic_tom')
    for el, kind in (('first_order_false_belief', 'false_belief'),):
        got = [s for s in sample(el, 300, seed=4808) if s.type == kind]
        df = pd.DataFrame({'question_text': [s.question for s in got], 'options': [s.options for s in got],
                           'domain': 'false-belief'})
        res = logic.solve_false_belief(df)
        ok = 0
        for s, (_, row) in zip(got, res.iterrows()):
            if row['status'] == 'ok':
                ok += 1
                assert row['pred'] == s.answer_index, (s.question, s.options)
        assert ok >= 0.9 * len(got), (el, ok, len(got))


def test_seeded_reproducibility():
    s = ElementSampler()
    for el in ('first_order_false_belief', 'second_order_false_belief'):
        for seed in range(15):
            assert s.sample_question(el, 7 + seed) == s.sample_question(el, 7 + seed)
