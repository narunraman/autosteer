import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from autosteer.steer_dataset import Dataset
from autosteer.paths import data_dir, templates_dir
from autosteer.filler_tools import function_registry
from autosteer.filler_tools import code_templates
from .template_fixes import apply_template_fixes, public_difficulty

MANIFEST_PATH: str | None = None   # set to write the manifest somewhere else (tests); default: in the data directory


def manifest_path() -> str:
    return os.path.abspath(MANIFEST_PATH or os.path.join(data_dir(), 'manifest.json'))
_PACKAGE = Path(__file__).resolve().parent.parent


_FINGERPRINT: dict = {}


def code_fingerprint() -> str:
    """Hash of the code that decides what the manifest says: the registered filler names and the sources of the
    sampler and filler packages (template fixes, hooks and code-written templates). A stored manifest with another
    fingerprint is rebuilt. Computed once per process (and set of fillers): the code a process runs does not change
    while it runs, and hashing every source took most of each load_manifest() call."""
    names = tuple(sorted(function_registry.keys()))
    if names not in _FINGERPRINT:
        _FINGERPRINT.clear()
        _FINGERPRINT[names] = _fingerprint(names)
    return _FINGERPRINT[names]


def _fingerprint(names) -> str:
    h = hashlib.sha256('\n'.join(names).encode())
    for sub in ('sampler', 'filler_tools'):
        for f in sorted((_PACKAGE / sub).rglob('*')):
            if f.suffix in ('.py', '.json') and '__pycache__' not in f.parts:
                h.update(f.relative_to(_PACKAGE).as_posix().encode() + b'\0' + f.read_bytes())
    return h.hexdigest()[:16]


def _template_names_on_disk() -> set[str]:
    """Elements with templates: a directory in the data directory's templates/, or code templates."""
    names = set(code_templates.elements()) | set(code_templates.composites())
    root = templates_dir()
    if not os.path.isdir(root):
        return names
    return names | {
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    }


def load_raw_templates(element_name: str) -> Dataset:
    """The element's templates before the load-time fixes: its code templates if it has any (they replace a
    template set in the data directory); for a composite element the concatenated corrected templates of its source
    elements (code_templates.register_composite); else templates/<element>/ of the data directory."""
    if code_templates.has_code_templates(element_name):
        return code_templates.dataset(element_name)
    if code_templates.is_composite(element_name):
        return code_templates.composite_dataset(
            element_name, lambda source: apply_template_fixes(source, load_raw_templates(source)))
    return Dataset.load(os.path.join(templates_dir(), element_name))


def _build_entry(element_name: str) -> dict:
    """Build a manifest entry from the template dataset (source of truth for sampling)."""
    dataset = apply_template_fixes(element_name, load_raw_templates(element_name))

    q = dataset.questions
    return {
        'domains': sorted(q['domain'].unique().tolist()),
        'types': sorted(q['type'].unique().tolist()),
        'difficulty_levels': sorted({str(public_difficulty(element_name, x)) for x in q['difficulty_level'].unique().tolist()}),
        'tags': sorted(q['tags'].unique().tolist()),
        'has_filler': element_name in function_registry,
        'template_question_count': len(q),
    }


def build_manifest(verbose: bool = True) -> dict:
    """
    Scan data/templates/, extract metadata from each Dataset, and write
    data/manifest.json. Returns the manifest dict.
    """
    element_names = sorted(_template_names_on_disk())
    manifest = {}

    for name in element_names:
        if verbose:
            print(f'  building entry: {name}')
        try:
            manifest[name] = _build_entry(name)
        except Exception as e:
            print(f'  WARNING: skipping {name} — {e}')

    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'code_fingerprint': code_fingerprint(),
        'elements': manifest,
    }

    manifest_path = globals()['manifest_path']()
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    # write-then-rename, so that parallel processes never read a half-written file; the temporary name is unique
    # to the thread as well, so two builds at once (the startup warm-up and a first request) do not share it
    tmp = f'{manifest_path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, manifest_path)

    if verbose:
        print(f'Manifest written to {manifest_path} ({len(manifest)} elements)')

    return output


def load_manifest(force_rebuild: bool = False) -> dict:
    """
    Load data/manifest.json, rebuilding it first if stale or missing.

    Staleness is defined as: the set of element directory names on disk no
    longer matches the set of keys in the stored manifest, or the stored
    manifest was built by other code (code_fingerprint).
    """
    if not force_rebuild and (stored := _current_manifest()) is not None:
        return stored
    # one build at a time in this process; a thread that waited finds the manifest the other one wrote
    with _BUILD_LOCK:
        if not force_rebuild and (stored := _current_manifest()) is not None:
            return stored
        return build_manifest(verbose=False)


_BUILD_LOCK = threading.Lock()


def _current_manifest() -> dict | None:
    """The stored manifest, or None if it is missing or stale."""
    manifest_path = globals()['manifest_path']()
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        stored = json.load(f)
    on_disk = _template_names_on_disk()
    if on_disk == set(stored['elements'].keys()) and stored.get('code_fingerprint') == code_fingerprint():
        return stored
    return None


def get_sampleable_elements(manifest: dict) -> list[str]:
    """Return element names that have both a template and a registered filler function."""
    return [
        name for name, entry in manifest['elements'].items()
        if entry['has_filler']
    ]
