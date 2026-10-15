"""One-off migration (October 2026): write each data-directory element's templates with the load-time corrections
(autosteer.sampler.template_fixes, consumer_template_fixes, behav_fixes) applied, so the corrections become part of
the data and the runtime no longer applies them.

    uv run python scripts/migrations/fold_template_fixes.py [--data-dir DIR]

Elements whose templates are written in code (code_templates) and composite elements are skipped: their data
folders, if any, are not read. The corrections are not idempotent, so this is run once, together with the change
that stops applying them at load. data_files.json is re-pinned afterwards (data_files.write_list).
"""
import argparse
import warnings

from autosteer import paths
from autosteer.filler_tools import code_templates
from autosteer.sampler.manifest import load_raw_templates
from autosteer.sampler.template_fixes import apply_template_fixes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args()
    if args.data_dir:
        paths.set_data_dir(args.data_dir)
    warnings.filterwarnings("ignore")
    root = paths.templates_dir()
    done, skipped = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        e = folder.name
        if code_templates.has_code_templates(e) or code_templates.is_composite(e):
            skipped.append(e)
            continue
        ds = apply_template_fixes(e, load_raw_templates(e))
        ds.save(folder)
        done.append(e)
    print(f"wrote {len(done)} elements; skipped {len(skipped)} (code templates): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
