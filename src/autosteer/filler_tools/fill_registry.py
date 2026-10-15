# src/filler_tools/fill_registry.py


def validate_filler_output(name: str, result: dict) -> None:
    """
    Validate the return value of a filler function.

    A filler function must return a dict. Two modes are supported:

    Mode A — filler provides options and answers:
        Required keys: 'options_list', 'correct_answer'
        - options_list : list of lists of option values, one inner list per sub-question.
                         e.g. [['12.5', '10.0', '8.3', '15.2']]
        - correct_answer: list of one-hot lists matching options_list.
                         e.g. [[1, 0, 0, 0]]
        Plus any number of additional string keys whose values fill {placeholder}
        slots in the question template.

    Mode B — filler provides only placeholder values (template defines options):
        Required keys: none beyond the placeholders expected by the template.
        The options and correct answers are taken directly from the template dataset.

    Raises ValueError with a descriptive message if the contract is violated.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"Filler '{name}' must return a dict, got {type(result).__name__}."
        )

    has_opts = 'options_list' in result
    has_ans = 'correct_answer' in result

    if has_opts ^ has_ans:
        raise ValueError(
            f"Filler '{name}': 'options_list' and 'correct_answer' must both be "
            f"present or both be absent."
        )

    if has_opts:
        opts = result['options_list']
        ans = result['correct_answer']

        if not isinstance(opts, list) or not opts:
            raise ValueError(
                f"Filler '{name}': 'options_list' must be a non-empty list."
            )
        if not isinstance(ans, list) or len(ans) != len(opts):
            raise ValueError(
                f"Filler '{name}': 'correct_answer' must be a list of the same "
                f"length as 'options_list' ({len(opts)})."
            )


class FillRegister:
    def __init__(self):
        self._functions = {}

    def register(self, name):
        def decorator(func):
            if name in self._functions:
                raise KeyError(f"Function '{name}' already in register.")
            self._functions[name] = func
            func._is_registered = True
            return func
        return decorator

    def get(self, name):
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not in register.")
        return self._functions[name]

    def keys(self):
        return self._functions.keys()

    @property
    def names(self):
        return list(self._functions.keys())

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, value):
        self._functions = value

    def __getitem__(self, name):
        return self._functions[name]

    def __contains__(self, name):
        return name in self._functions


    