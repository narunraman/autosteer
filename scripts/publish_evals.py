"""Publish merged Evals results to the site, holding back runs that are not ready.

Writes src/autosteer/web/static/evals_results.json from a full merge (steer-scoring merge_inspect.py output),
keeping only the (model, benchmark) results that the Evals pages show: open models whose release run has
multiple-choice scores on at least site.EVAL_LISTED_SHARE of the benchmark's elements, and closed models with
any. Held-back runs stay in the merge and appear here once they are complete.

    uv run python scripts/publish_evals.py merged.json
"""
import json
import sys

from autosteer.web import site

results = json.loads(open(sys.argv[1], encoding="utf-8").read())
out = site.publishable_eval_results(results)
site.EVAL_RESULTS_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
for v in site.SLUGS:
    shown = [m["name"] for m in out["models"] if v in m["data"]]
    print(f"{v}: {len(shown)} models shown")
held = sorted({m["name"] for m in results["models"]} - {m["name"] for m in out["models"]})
print(f"held back entirely ({len(held)}): {', '.join(held)}")
print("wrote", site.EVAL_RESULTS_PATH)
