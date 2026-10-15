# src/steer_generation/directory_utils.py

# Built-in packages
import os
import re

# Get the base directory of the project
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# The template data: AUTOSTEER_DATA_DIR if set, else data/ of this checkout (the repository holds it), else
# ~/.autosteer/data (an installed package without the repository).
REPO_DATA_PATH = os.path.join(BASE_PATH, 'data')
DEFAULT_DATA_PATH = REPO_DATA_PATH if os.path.isdir(os.path.join(REPO_DATA_PATH, 'templates')) else os.path.expanduser('~/.autosteer/data')
DATA_PATH = os.path.abspath(
    os.environ.get(
        'AUTOSTEER_DATA_DIR',
        os.environ.get('STEER_DATA_DIR', DEFAULT_DATA_PATH),
    )
)

EXAMPLE_SEEDS_DIR = 'example_seeds'
GENERATED_SEEDS_DIR = 'generated_seeds'
TEMPLATES_DIR = 'templates'
ELEMENTS_DIR = 'elements'

EXAMPLE_SEEDS_PATH = os.path.join(DATA_PATH, EXAMPLE_SEEDS_DIR)
GENERATED_SEEDS_PATH = os.path.join(DATA_PATH, GENERATED_SEEDS_DIR)
TEMPLATES_PATH = os.path.join(DATA_PATH, TEMPLATES_DIR)
ELEMENTS_PATH = os.path.join(DATA_PATH, ELEMENTS_DIR)

def get_data_directory(dir_type):
    if dir_type == 'example_seeds':
        dir = EXAMPLE_SEEDS_DIR
    elif dir_type == 'generated_seeds':
        dir = GENERATED_SEEDS_DIR
    elif dir_type == 'templates':
        dir = TEMPLATES_DIR
    elif dir_type == 'elements':
        dir = ELEMENTS_DIR
    else:
        raise ValueError(f"Invalid dir_type: {dir_type}. Only the following dir_types are supported: 'example_seeds', 'generated_seeds', 'templates', and 'elements'.")

    return dir

def _remove_other_base_dir(dir, dir_type):
    if dir_type == 'example_seeds':
        return re.sub(rf"({GENERATED_SEEDS_DIR}|{TEMPLATES_DIR}|{ELEMENTS_DIR})", EXAMPLE_SEEDS_DIR, dir)
    elif dir_type == 'generated_seeds':
        return re.sub(rf"({EXAMPLE_SEEDS_DIR}|{TEMPLATES_DIR}|{ELEMENTS_DIR})", GENERATED_SEEDS_DIR, dir)
    elif dir_type == 'templates':
        return re.sub(rf"({EXAMPLE_SEEDS_DIR}|{GENERATED_SEEDS_DIR}|{ELEMENTS_DIR})", TEMPLATES_DIR, dir)
    elif dir_type == 'elements':
        return re.sub(rf"({EXAMPLE_SEEDS_DIR}|{GENERATED_SEEDS_DIR}|{TEMPLATES_DIR})", ELEMENTS_DIR, dir)
    else:
        raise ValueError(f"Invalid dir_type: {dir_type}. Only the following dir_types are supported: 'example_seeds', 'generated_seeds', 'templates', and 'elements'.")
    
def get_data_directory_path(dir_type, append_dir=None):
    """
    Get the path of the data directory based on the dir_type.

    ## Args:
    **dir_type:** (str) The type of directory. It can be 'example_seeds', 'generated_seeds', or 'templates'.
    **append_dir:** (str) The element directory to append to the base directory. Default is None.

    """
    if dir_type == 'example_seeds':
        dir = EXAMPLE_SEEDS_PATH
    elif dir_type == 'generated_seeds':
        dir = GENERATED_SEEDS_PATH
    elif dir_type == 'templates':
        dir = TEMPLATES_PATH
    elif dir_type == 'elements':
        dir = ELEMENTS_PATH
    else:
        raise ValueError(f"Invalid dir_type: {dir_type}. Only the following dir_types are supported: 'example_seeds', 'generated_seeds', 'templates', and 'elements'.")

    if append_dir is not None:
        dir = os.path.join(dir, append_dir)
        dir = _remove_other_base_dir(dir, dir_type=dir_type)

    return dir

def dir_exists(dir_type, append_dir=None):
    return os.path.exists(get_data_directory_path(dir_type, append_dir))

def get_files(dir_type):
    return os.listdir(get_data_directory_path(dir_type))
