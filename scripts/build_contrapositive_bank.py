"""Build the equiv_contrapositive context bank (October 2026 cleanup).

Input: the 560 contexts recovered from the released GPT-4 questions (data directory,
templates/equiv_contrapositive/template_labels.csv; audit/steer/regen/recover_contrapositive_templates.py in
STEER-generation). Each has the statement's clauses P and Q and GPT-written negations not-P / not-Q taken from one of
the question's options, and those negations often differ in style from the statement ("she goes to bed early" /
"she did not go to bed early" / "Sarah does not wake up feeling refreshed"), so the old options could give the key away.

Here every context is checked and normalised by rule; the GPT negations are used only to locate the verb and its
base form, never as text:
  - contractions expanded (it's -> it is, doesn't -> does not);
  - generic rules: an indefinite subject of P ("a patient", "someone") with a pronoun subject of Q ("they", "it",
    "its") becomes the same definite subject in both ("the patient has X" / "the patient has Y"), with the verb of
    Q made to agree, so that no option starts with a pronoun whose noun comes later;
  - not-P and not-Q are built from P and Q themselves, in their own tense and with their own subject: "AUX not"
    for is / are / was / will / has + participle, "does not" / "do not" + base form for a simple present;
    "there is not (a) X" -> "there is no X"; a stray "also" in Q is dropped.
Dropped (flag printed): a clause already negative or with and / or / but / unless (no clean negation), no
recognised verb pattern, past tense, an indefinite subject referred to by a pronoun that could not be rewritten,
first or second person (I, you, we: the options would be read outside the quote), P and Q sharing content words,
and the contexts in DROP (read by hand). The question shows "If P, then Q" from the same normalised clauses, and the
filler builds every option from them, so the key and the distractors share their wording exactly.

    python scripts/build_contrapositive_bank.py <data dir>/templates/equiv_contrapositive/template_labels.csv \
        src/autosteer/filler_tools/contrapositive_bank.json
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys

# read by hand: the pronoun in Q refers to the action, not to a subject ("it will minimize his investment risks")
DROP = {'ec0559': 'Q starts with "it" referring to the diversification, not to a subject'}

CONTR = [(r"\bit's\b", 'it is'), (r"\bdoesn't\b", 'does not'), (r"\bdon't\b", 'do not'), (r"\bdidn't\b", 'did not'),
         (r"\bisn't\b", 'is not'), (r"\baren't\b", 'are not'), (r"\bwon't\b", 'will not'), (r"\bhasn't\b", 'has not'),
         (r"\bhaven't\b", 'have not'), (r"\bwasn't\b", 'was not'), (r"\bweren't\b", 'were not'), (r"\bthey're\b", 'they are'),
         (r"\bhe's\b", 'he is'), (r"\bshe's\b", 'she is'), (r"\bI'm\b", 'I am'), (r"\byou're\b", 'you are')]
AUX = {'is', 'are', 'am', 'was', 'were', 'will', 'has', 'have', 'had'}
PRON = {'i', 'you', 'we', 'they', 'he', 'she', 'it'}
PLURAL_SUBJ = {'i', 'you', 'we', 'they'}
REFER = re.compile(r"\b(they|them|their|it|its|he|she|his|her|him)\b", re.I)
FIRST_SECOND = re.compile(r"\b(I|me|my|we|our|us|you|your)\b")
INDEF = re.compile(r"^(a|an|someone|somebody|anyone|any)\b", re.I)
STOP = set('a an the of to in on for at by with and or is are was were be been has have had will not do does did '
           'that this their they them it its he she his her him i you we my your our more less than very'.split())


def expand(s):
    for a, b in CONTR:
        s = re.sub(a, b, s, flags=re.I) if a != r"\bI'm\b" else re.sub(a, b, s)
    return s.strip()


def inflect3(lemma):
    irregular = {'have': 'has', 'do': 'does', 'go': 'goes', 'be': 'is'}
    if lemma in irregular:
        return irregular[lemma]
    if re.search(r'(s|sh|ch|x|z|o)$', lemma):
        return lemma + 'es'
    if re.search(r'[^aeiou]y$', lemma):
        return lemma[:-1] + 'ies'
    return lemma + 's'


def deinflect(verb):
    irregular = {'has': 'have', 'does': 'do', 'goes': 'go'}
    if verb in irregular:
        return irregular[verb]
    if not verb.endswith('s') or verb.endswith('ss'):
        return None
    if re.search(r'[^aeiou]ies$', verb):
        return verb[:-3] + 'y'
    if re.search(r'(sh|ch|ss|x|o)es$', verb):
        return verb[:-2]
    return verb[:-1]


DEF = {'someone': 'the person', 'somebody': 'the person', 'anyone': 'the person', 'a person': 'the person'}


def definite(P, Q):
    """'a patient has X' / 'they have Y' -> 'the patient has X' / 'the patient has Y'. Returns (P, Q, note)."""
    m = re.match(r'^(someone|somebody|anyone|(?:a|an) [\w-]+(?: [\w-]+)??)\s+(is|are|has|have|will|\w+s)\b', P)
    if not m:
        return P, Q, None
    np_ = m.group(1)
    if np_.lower() in DEF:
        new = DEF[np_.lower()]
    elif re.match(r'^(a|an) ', np_):
        # the subject NP must end before the verb: 'a blood sample tests' -> np 'a blood sample'
        new = 'the ' + np_.split(' ', 1)[1]
    else:
        return P, Q, None
    qt = Q.split()
    if qt[0] in ('they', 'it'):
        verb = qt[1]
        if qt[0] == 'they':
            agree = {'are': 'is', 'have': 'has', 'were': 'was', 'do': 'does'}
            if verb in agree:
                verb = agree[verb]
            elif verb not in ('will',) and not verb.endswith('ed'):
                verb = inflect3(verb)
        Q2 = ' '.join([new, verb] + qt[2:])
    elif qt[0] in ('their', 'its'):
        Q2 = ' '.join([new + "'s"] + qt[1:])
    else:
        return P, Q, None
    return new + P[len(np_):], Q2, f'definite: {np_} -> {new}'


def negate(pos, neg_candidates):
    """Negation of clause `pos` (subject + verb phrase), built from pos's own subject and tense; the GPT negations
    only tell where the verb is and its base form. Returns (negation, reason) with negation None if not clean."""
    toks = pos.split()
    low = [t.lower() for t in toks]
    if 'not' in low or 'never' in low or 'no' in low or "n't" in pos:
        return None, 'already negative'
    if re.search(r'\b(and|or|but|either|neither|unless)\b', pos):
        return None, 'conjunction'
    # 1) auxiliary in pos followed by a participle / -ing / adjective, with a candidate "AUX not ..."
    for c in neg_candidates:
        c = expand(c)
        m = re.search(r'\b(is|are|am|was|were|will|has|have|had) not (.+)$', c)
        if m:
            rest = m.group(2)
            mm = re.search(r'\b(is|are|am|was|were|will|has|have|had) ' + re.escape(rest) + '$', pos)
            if mm:
                aux = mm.group(1)
                if aux in ('has', 'have', 'had') and not re.match(r'(been|\w+ed|\w+en)\b', rest):
                    continue  # "has a project": main verb, needs do-support
                return pos[:mm.start()] + f'{aux} not {rest}', 'aux'
        m = re.search(r'\b(?:does|do|did) not (\w+)(.*)$', c)
        if m:
            lemma, rest = m.group(1), m.group(2)
            mm = re.search(r'(^|\s)(\w+)' + re.escape(rest) + '$', pos)
            if not mm:
                continue
            verb = mm.group(2)
            subj = pos[:mm.start(2)].strip()
            if not subj:
                continue
            if subj.split()[-1] == 'will' and verb == lemma:
                return f'{subj} not {lemma}{rest}', 'will'
            head = subj.split()[-1].lower()
            if verb == lemma and (head in PLURAL_SUBJ or (head.endswith('s') and not head.endswith('ss'))):
                return f'{subj} do not {lemma}{rest}', 'do'
            if verb == inflect3(lemma) or (lemma == 'have' and verb == 'has'):
                return f'{subj} does not {lemma}{rest}', 'does'
            continue
    # 2) no usable candidate: a simple-present verb right after a one-word subject (pronoun or name)
    if len(toks) >= 2 and (low[0] in PRON or (toks[0][0].isupper() and toks[0].isalpha() and low[0] not in STOP)):
        verb = low[1]
        if low[0] in PLURAL_SUBJ and verb not in AUX:
            return f'{toks[0]} do not {verb} ' + ' '.join(toks[2:]), 'own-do'
        lemma = deinflect(verb)
        if lemma and verb not in AUX:
            return f'{toks[0]} does not {lemma} ' + ' '.join(toks[2:]), 'own-does'
    # 3) plain auxiliary in pos
    m = re.match(r'^(.+?) (is|are|am|was|were|will) (.+)$', pos)
    if m and m.group(1).lower() not in ('',):
        return f'{m.group(1)} {m.group(2)} not {m.group(3)}', 'aux-own'
    return None, 'no negation pattern'


def subject(clause, neg):
    """subject = the part of clause before the verb (the negation differs from the clause right after it)."""
    a, b = clause.split(), neg.split()
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if b[i] in ('does', 'do'):
        return ' '.join(a[:i])
    return ' '.join(a[:i - 1]) if b[i] == 'not' else ' '.join(a[:i])


def analyse(r):
    flags = []
    P, Q = expand(r['P']), expand(r['Q'])
    P, Q, note = definite(P, Q)
    if note:
        flags_note = note
    nP, why_p = negate(P, [r['notP'], r['inv_notP']])
    nQ, why_q = negate(Q, [r['notQ'], r['inv_notQ']])
    if nP is None:
        flags.append('negation P: ' + why_p)
    if nQ is None:
        flags.append('negation Q: ' + why_q)
    if nP and nQ:
        sp, sq = subject(P, nP), subject(Q, nQ)
        if (INDEF.match(sp) or sp.lower().startswith('the ')) and REFER.search(Q) and not note:
            flags.append('pronoun in Q refers to an indefinite subject of P')
        if INDEF.match(sq) and REFER.search(P):
            flags.append('pronoun in P refers to the subject of Q')
        tp = 'future' if re.search(r'\bwill\b', P) else 'past' if re.search(r'\b(was|were|did|had)\b|\w+ed\b', P.split(' ', 2)[1] if len(P.split()) > 1 else '') else 'present'
        tq = 'future' if re.search(r'\bwill\b', Q) else 'present'
        if tp == 'future':
            flags.append('future tense in the antecedent')
        if re.search(r'\b(was|were|did|had)\b', P + ' ' + Q):
            flags.append('past tense')
    if FIRST_SECOND.search(P + ' ' + Q):
        flags.append('first or second person')
    cw = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}
    shared = cw(P) & cw(Q)
    names = set(re.findall(r'\b[A-Z][a-z]+\b', r['intro'] + ' ' + P + ' ' + Q))
    shared -= {n.lower() for n in names}
    if note:
        shared -= set(note.split(' -> ')[1].lower().split())
    if shared:
        flags.append('shared words: ' + ','.join(sorted(shared)))
    return dict(P=P, Q=Q, notP=nP, notQ=nQ, flags=flags, note=note)



def tidy(clause: str, is_q: bool) -> str:
    clause = re.sub(r'\bthere (is|was) not (?:a |an )?', r'there \1 no ', clause)
    if is_q:
        clause = re.sub(r' also\b', '', clause)
    return re.sub(r'\s+', ' ', clause).strip()


def build(rows):
    bank, dropped = [], collections.Counter()
    for r in rows:
        a = analyse(r)
        if r['template_id'] in DROP:
            a['flags'].append('hand: ' + DROP[r['template_id']])
        if a['flags']:
            for f in a['flags']:
                dropped[f.split(':')[0]] += 1
            continue
        P, Q = tidy(a['P'], False), tidy(a['Q'], True)
        nP, nQ = tidy(a['notP'], False), tidy(a['notQ'], True)
        bank.append(dict(id=r['template_id'], domain=r['domain'], intro=r['intro'], if_word=r['if_word'],
                         outro=r['outro'], P=P, Q=Q, notP=nP, notQ=nQ))
    return bank, dropped


if __name__ == '__main__':
    rows = list(csv.DictReader(open(sys.argv[1], newline='')))
    bank, dropped = build(rows)
    # one context per distinct question text
    seen, out = set(), []
    for b in bank:
        key = (b['intro'], b['if_word'], b['P'], b['Q'], b['outro'])
        if key not in seen:
            seen.add(key)
            out.append(b)
    print(f'{len(rows)} contexts, {len(out)} kept ({len(bank) - len(out)} duplicates after normalising)')
    print('dropped (a context can have several flags):', dict(dropped.most_common()))
    print('kept by domain:', dict(collections.Counter(b['domain'] for b in out)))
    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write('\n')
