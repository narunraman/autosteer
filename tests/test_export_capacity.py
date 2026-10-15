"""Export capacity: the exporter fills every cell of these elements to its target without duplicates.

The export rule (release/export.py) asks for 20 questions per (type, domain, difficulty) cell, or the floor of 200
spread over the cells, and skips draws that repeat an earlier question. conditional_syllogism (numbers cells) and
prob_basic (compute cells for cards, coins and dice) used to fall short; templates and events were added. The other
elements here were rebuilt in the same cleanup (October 2026). This runs
the exporter's own per-element routine with the default rule and the default seed and requires no shortfall.
"""
import pytest

from autosteer.release.export import DEFAULT_SEED, export_element

ELEMENTS = ['conditional_syllogism', 'prob_basic', 'equiv_contrapositive', 'dsic_mechanism', 'constrained_optimization',
            'input_price_response']


# Cells whose question text alone must differ between questions (not only the distractors): every cell of
# conditional_syllogism (its options are fixed by the template) and the compute cells of prob_basic.
DISTINCT_TEXT = {'conditional_syllogism': lambda r: True, 'prob_basic': lambda r: r['type'] == 'compute'}


@pytest.mark.parametrize('element', ELEMENTS)
def test_every_cell_reaches_its_target(element):
    out = export_element(element, DEFAULT_SEED, 20, 200, 20, 'test')
    st = out['stats']
    assert st['shortfall'] == 0 and not st['short_cells'], st
    assert st['questions'] >= 200
    if element in DISTINCT_TEXT:
        rows = [r for r in out['rows'] if DISTINCT_TEXT[element](r)]
        assert rows and len({r['question_text'] for r in rows}) == len(rows)
