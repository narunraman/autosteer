# src/filler_tools/__init__.py

# Built-in packages
import os
import importlib
import traceback
# Local packages
from .fill_registry import FillRegister
function_registry = FillRegister()


# Global variable to track if functions have been registered
_functions_registered = False

def _import_and_register_functions():
    """Dynamically imports modules in this package directory and registers functions."""
    global _functions_registered

    if _functions_registered:
        return
    
    package_dir = os.path.dirname(__file__)

    for filename in os.listdir(package_dir):
        if filename.endswith('_tools.py') and not filename.startswith('_'):
            modulename = filename.split('.')[0]  # Filename without the extension.
            package_module = '.'.join([__name__, modulename])
            try:
                module = importlib.import_module(package_module)
                if hasattr(module, 'function_registry'):
                    for name, func in module.function_registry.functions.items():
                        function_registry.functions[name] = func
            except Exception:
                traceback.print_exc()
                raise
    # print('Finished Registering functions')
    _functions_registered = True

# Build the registry
_import_and_register_functions()