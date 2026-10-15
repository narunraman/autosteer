
# Creating a New `_tools.py` File

## Guidelines
When creating a new `_tools.py` file, you must follow these guidelines:

1. **File Naming**: The file name must end with `_tools.py` and should not start with an underscore `_`.

2. **Defining Functions**: A fill function `fill_function` can only take a single input: `metadata`. The metadata is the relevant information about the question and should uniquely define the numerical values within the text. The function must return a dictionary of key value pairs where the keys are the format strings in the question. For example for the following question (in display_json format):

    ```json
    {
        "question_1": "What is the capital of {country}?",
        "options_1": [
            "{capital_1}",
            "{capital_2}",
            "{capital_3}",
            "{correct_capital}"
        ]
        "answer_1": "{correct_capital}"
    }
    ```
    The function must return the following:
    ```json
    {
        "country": "France",
        "capital_1": "Suva",
        "capital_2": "Sacramento",
        "capital_3": "Germany",
        "correct_capital": "Paris"
    }
    ```

    In the case where the fill function must also supply options and/or a correct answer, then the return value must contain an `options_list` and/or a `correct_answer` key. The `options_list` key must be a list of lists where each sub-list is the options for a sub-question. The `correct_answer` key must be a list of correct answers or a list of boolean masks. In the first case the shape of the `correct_answer` list must be the same as the number of sub-questions and in the second the shape must be the same as `options_list`. For example, in the first case:
    ```json
    {
        "var_1": "something",
        "var_2": "something_else",
        "options_list": [
            ["option_1_1", "option_1_2"],
            ["option_2_1", "option_2_2"]
        ],
        "correct_answer": [
            "option_1_2", 
            "option_2_1"
        ]
    }


3. **Registering Functions**: Every fill function must be registered.
    1. Instantiate a module-level function registry, using the [`FillRegister`](fill_registry.py "Go to definition") class to register your functions. The package-level registry imports every `_tools.py` module and collects each module's `function_registry`. It is important that you define the module register as `function_registry`.
    2. Use the register decorator on each fill function.

    ### Example:
    ```python
    from .fill_registry import FillRegister
    function_registry = FillRegister()

    @function_registry.register('my_function')
    def my_fill_function(metadata):
        # Function implementation
        pass
    ```

4. **Avoid Name Conflicts**: Ensure that the function names you register do not conflict with existing function names in the package-level `function_registry`. If a conflict is detected inside a module, an error will be raised. To inspect the currently registered names, import `function_registry` from `autosteer.filler_tools` and check `function_registry.names`.


5. Some style guidelines: 

- **Import Statements**: Organize your import statements into three sections: built-in packages, external packages, and local packages. For example:

    ```python
    # Built-in packages
    import random
    import re

    # External packages
    import numpy as np

    # Local packages
    from .fill_registry import function_registry
    ```
- **Utility Functions**: If there are any utility functions that you would like to keep in a different file, place them into the `utilities/` directory. At the top of your `_tools.py` file import `from .utilities.your_utility_function import your_function`.

## Randomness


Import the random number generator from `.utilities.filler_utils`
For reproducibility, generate templates from a freshly constructed instance. 

## Example

Here is an example of a new `awesome_tools.py` file:

```python
# Built-in packages
import os

# External packages
import pandas as pd

# Local packages
from .fill_registry import FillRegister
function_registry = FillRegister()
from .utilities.filler_utils import rng

@function_registry.register('example_function')
def example_function(metadata):
    """
    An example function that performs a specific task.
    """
    values = {}
    if metadata['domain'] == 'education':
        values['first_num'] = rng.random()
    if metadata['difficulty_level'] == 4:
        values['first_num'] = [values['first_num'], rng.random()]
    return values
