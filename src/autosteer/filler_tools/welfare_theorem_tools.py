

from .utilities.filler_utils import gen_answer_and_options
from . fill_registry import FillRegister
function_registry = FillRegister()


@function_registry.register('welfare_theorem_1')
def gen_welfare_theorem_1(metadata):
    # 'pareto_efficient' variants ask whether the CE allocation is Pareto
    # efficient (yes); the original questions ask whether further mutually
    # beneficial trades exist (no).
    # With lump-sum transfers the efficient allocation is supported by CE
    # prices (yes); without redistribution it is not guaranteed (no).
    options = ['yes', 'no']
    answer = 'no' if metadata.get('type') == 'without_transfers' else 'yes' if metadata.get('type') == 'pareto_efficient' else 'no'
    answer_mask, options = gen_answer_and_options(answer, options=options)
    return {
        "options_list": options,
        "correct_answer": answer_mask
    }

@function_registry.register('welfare_theorem_2')
def gen_welfare_theorem_2(metadata):
    # With lump-sum transfers the efficient allocation is supported by CE
    # prices (yes); without redistribution it is not guaranteed (no).
    options = ['yes', 'no']
    answer = 'no' if metadata.get('type') == 'without_transfers' else 'yes'
    answer_mask, options = gen_answer_and_options(answer, options=options)
    return {
        "options_list": options,
        "correct_answer": answer_mask
    }
