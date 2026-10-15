"""Recover the decoy-effect (independence) scenario templates (October 2026).

Source: the v1 template pickle preferences_gen/decoy.pkl (4,780 GPT-4 scenario sentences, one `{}` slot for the
number of options), kept in the 2024 data-generation archive (not public). The released
element used the everyday-products and technological-products scenarios, with "two" in part 0 and "three" in part 1.
The option generator (fill_data/.../decoy_dollars.pkl) is lost; autosteer generates the options
(filler_tools/decoy_tools.py).

Kept: scenarios of those two domains with exactly one `{}` that is a number of options (not after "a" / "an"),
that name no attribute other than price and quality (the options show only a price and a quality rating), one per
distinct sentence.

    python scripts/recover_decoy_templates.py <decoy.pkl> src/autosteer/filler_tools/decoy_templates.json
"""
from __future__ import annotations

import collections
import json
import re
import sys

import pandas as pd

DOMAINS = {'everyday products': 'everyday_products', 'technological products': 'technological_products'}
OTHER_ATTRIBUTES = re.compile(
    r'feature|size|power|setting|design|spec|function|capacit|battery|storage|colou?r|speed|style|material|'
    r'brand reputation|warrant|weight|screen|resolution|camera|memory|wattage|energy|efficien|performance|'
    r'option[s]? for|level|setting|mode|attachment|accessor|durab|comfort|noise|flavou?r|portab', re.I)


def main(src: str, out: str) -> None:
    d = pd.read_pickle(src)
    stats, kept, seen = collections.Counter(), [], set()
    for i, r in d.iterrows():
        q = re.sub(r'\s+', ' ', str(r.questions)).strip()
        if r.domain not in DOMAINS:
            stats['domain'] += 1
            continue
        if q.count('{}') != 1 or re.search(r'\{\w+\}', q):
            stats['slots'] += 1
            continue
        if re.search(r'\b(a|an|the)\s+\{\}', q, re.I):
            stats['article'] += 1
            continue
        if OTHER_ATTRIBUTES.search(q):
            stats['other_attributes'] += 1
            continue
        if q in seen:
            stats['duplicate'] += 1
            continue
        seen.add(q)
        stats['ok'] += 1
        kept.append({'source_row': int(i), 'domain': DOMAINS[r.domain], 'context': r.type == 'context', 'text': q})
    doc = {'about': __doc__.strip().splitlines()[0], 'source': src, 'counts': dict(stats), 'templates': kept}
    with open(out, 'w') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(dict(stats))


if __name__ == '__main__':
    main(*sys.argv[1:3])
