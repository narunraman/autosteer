"""Quality audit of generated questions: sample N per released element and flag text defects.

Usage: python audit.py OUT_DIR [N] [element ...]
Writes OUT_DIR/flags.json (per element counts and examples) and OUT_DIR/samples.jsonl.
"""
import json
import re
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

N = 40
PH = re.compile(r'\{[A-Za-z_][A-Za-z0-9_]*\}')
NEGZERO = re.compile(r'(?<![\w.])-0(?:\.0+)?(?![\d.])')
ZERO_DEC = re.compile(r'(?<![\w.\-])0\.0+(?![\d])')
ZERO_INT = re.compile(r'(?<![\w.\-,$])0(?![\d.,%:])(?!\s*(?:and|or|to|-|<|≤)\b)')
BADVAL = re.compile(r'\b(nan|NaN|inf|Infinity)\b|\bNone\b(?! of)')
SCI = re.compile(r'\de[+-]\d')
TINY = re.compile(r'(?<![\d.])-?0\.0\d(?!\d)')
TABLE = re.compile(r'^\s|\S {2,}\S.*\S {2,}\S|^\s*(Action|Type|Strategy)\b.*\(|^[\w ]+\s+\(-?[\d.]+, -?[\d.]+\)')
VERB = re.compile(r'\b(determine|calculate|find|compute|identify|select|choose|estimate|derive|what|which|how|evaluate|solve|express|give|state|rank|indicate)\b', re.I)
LONGDEC = re.compile(r'\d\.\d{5,}')
REPR = re.compile(r"\('[^']*', |\['|'\]|\[\(|np\.|array\(|Decimal\(|Fraction\(|\{'|\"[a-z_]+\":|<[a-z_.]+ object")
DSPACE = re.compile(r'(?<=\S)  +(?=\S)')
NOSPACE = re.compile(r'(?<![A-Z.])\b[a-z]{2,}[,;:][A-Za-z]|[a-z]{3,}[.!?][A-Z][a-z]')
SPACEPUNCT = re.compile(r'\w \s*[,.;?!](?!\d)')
END_OK = re.compile(r'(\?|:)\s*$')
INSTR = re.compile(r'^\s*(Select|Choose|Pick|Find|Calculate|Compute|Determine|Identify|Give|State|What|Which|How|Rank|Indicate|'
                   r'Answer|Is|Are|Does|Do|Did|Should|Would|Will|Can|Please|Estimate|Derive|Express|Round|Order|List|Decide|Evaluate|Solve|'
                   r'Based on|Using|Given|According|Who|Where|When|Assuming|Tell|Report|Name|Classify|Consider)', re.I)
NUM = re.compile(r'^\s*[-+]?\$?\s*-?[\d,]*\.?\d+(?:e[-+]?\d+)?\s*%?\s*$')


def num(s):
    if not NUM.match(s):
        return None
    try:
        return float(s.replace('$', '').replace(',', '').replace('%', '').strip())
    except ValueError:
        return None


def last_sentence(text):
    t = text.strip()
    parts = re.split(r'(?<=[.!?])\s+|\n+', t)
    parts = [p for p in parts if p.strip()]
    return parts[-1] if parts else ''


REPOSS = re.compile(r"\b([A-Z][a-z]+)\b(?=[^.?!\n]*?\b\1's\b)")
NOT_NAME = {'The', 'If', 'In', 'A', 'An', 'Each', 'Both', 'This', 'That', 'When', 'What', 'Which', 'How', 'At', 'For', 'Action'}


def repeated_possessive(text):
    for m in REPOSS.finditer(text):
        if m.group(1) not in NOT_NAME:
            return m
    return None


def prose(text):
    return '\n'.join(l for l in text.split('\n')
                     if not TABLE.search(l) and len(re.findall(r'(?<=\S) {2,}(?=\S)', l)) < 2)


def check(q, consistency):
    flags = []
    for pi, part in enumerate(q['parts']):
        text, opts = part['question'], part['options']
        alltext = [prose(text)] + list(opts)
        for s in alltext:
            for name, rx in (('placeholder', PH), ('neg_zero', NEGZERO), ('zero_decimal', ZERO_DEC),
                             ('bad_value', BADVAL), ('long_decimal', LONGDEC), ('sci_notation', SCI), ('tiny_value', TINY), ('repr', REPR),
                             ('double_space', DSPACE), ('missing_space', NOSPACE), ('space_before_punct', SPACEPUNCT)):
                m = rx.search(s)
                if m:
                    flags.append((name, pi, s[max(0, m.start() - 50): m.end() + 50]))
            m = repeated_possessive(s)
            if m:
                flags.append(('repeated_name_possessive', pi, s[max(0, m.start() - 20): m.end() + 120]))
            m = ZERO_INT.search(s)
            if m:
                flags.append(('zero_int', pi, s[max(0, m.start() - 50): m.end() + 50]))
        if len(set(opts)) < len(opts):
            flags.append(('dup_options', pi, ' | '.join(opts)))
        vals = [num(o) for o in opts]
        nv = [v for v in vals if v is not None]
        if len(nv) == len(opts) and len(set(nv)) < len(nv) and len(set(opts)) == len(opts):
            flags.append(('equal_numeric_options', pi, ' | '.join(opts)))
        elif len(nv) >= 2:
            s = sorted(nv)
            for a, b in zip(s, s[1:]):
                if a != b and abs(a - b) <= 0.005 * max(abs(a), abs(b)):
                    flags.append(('near_equal_options', pi, ' | '.join(opts)))
                    break
        nc = sum(part['correct'])
        if not consistency and nc != 1:
            flags.append(('correct_count_%d' % nc, pi, ' | '.join(opts)))
        pt = prose(text).rstrip()
        ls = last_sentence(pt)
        if opts and not END_OK.search(pt) and not INSTR.match(ls) and not VERB.search(ls):
            flags.append(('no_question_end', pi, pt[-160:]))
    return flags


def audit_element(args):
    element, n, consistency = args
    from autosteer.sampler.generator import ElementSampler
    s = ElementSampler()
    out = {'element': element, 'flags': defaultdict(list), 'errors': [], 'samples': []}
    try:
        tmpl = s._load_template(element)
        qf, _ = s._pool_frames(tmpl, 'questions')
        types = sorted(set(str(t) for t in qf['type'].dropna().unique()))
    except Exception as e:
        out['errors'].append('load: %r' % e)
        types = []
    for i in range(n):
        seed = 7001 + i
        try:
            if types and i % 2 == 0:
                q = s.sample_question(element, seed, type=types[(i // 2) % len(types)])
            else:
                q = s.sample_question(element, seed)
        except Exception as e:
            out['errors'].append('seed %d: %s' % (seed, ''.join(traceback.format_exception_only(type(e), e)).strip()[-300:]))
            continue
        out['samples'].append(q)
        for name, pi, ex in check(q, consistency):
            out['flags'][name].append({'seed': seed, 'template': q['template_id'], 'type': str(q['metadata']['type']),
                                       'part': pi, 'ex': ex})
    out['flags'] = dict(out['flags'])
    return out


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N
    root = Path(__file__).resolve().parent
    els = json.load(open(Path.home() / 'UBC/autosteer-wt/quality/src/autosteer/web/content/elements.json'))
    released = {e['element']: e['grading'] == 'consistency' for e in els if e['status'] != 'held'}
    want = sys.argv[3:] or sorted(released)
    jobs = [(e, n, released.get(e, False)) for e in want]
    summary = {}
    with Pool(12) as p, open(outdir / 'samples.jsonl', 'w') as f:
        for r in p.imap_unordered(audit_element, jobs):
            for q in r.pop('samples'):
                f.write(json.dumps(q, default=str) + '\n')
            summary[r['element']] = r
            print(r['element'], {k: len(v) for k, v in r['flags'].items()}, len(r['errors']), flush=True)
    json.dump(summary, open(outdir / 'flags.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
