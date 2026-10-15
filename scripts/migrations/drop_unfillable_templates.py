"""One-off migration (October 2026): remove from templates/ the templates whose text uses a placeholder the element's
filler never returns (they were dropped at load by ElementSampler._drop_unfillable). tests/test_template_placeholders.py
checks the stored templates from now on."""
import warnings

from autosteer import paths
from autosteer.filler_tools import code_templates
from autosteer.sampler import ElementSampler
from autosteer.sampler.manifest import load_raw_templates


def main() -> None:
    warnings.filterwarnings("ignore")
    s = ElementSampler(seed=0)
    changed = []
    for folder in sorted(p for p in paths.templates_dir().iterdir() if p.is_dir()):
        e = folder.name
        if code_templates.has_code_templates(e) or code_templates.is_composite(e) or e not in s._sampleable:
            continue
        stored, kept = load_raw_templates(e), s._load_template(e)
        if len(stored.questions) == len(kept.questions) and len(stored.example_questions) == len(kept.example_questions):
            continue
        before = (stored.questions['question_id'].nunique(), stored.example_questions['question_id'].nunique())
        for attr in ('questions', 'options', 'example_questions', 'example_options'):
            setattr(stored, attr, getattr(kept, attr))
        stored.save(folder)
        changed.append(f"{e}: {before} -> {(kept.questions['question_id'].nunique(), kept.example_questions['question_id'].nunique())}")
    print("\n".join(changed) or "nothing to remove")


if __name__ == "__main__":
    main()
