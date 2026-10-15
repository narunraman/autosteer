"""The web app's routes, in order, and its OpenAPI schema are unchanged (a guard for restructuring web/).

    uv run python tests/test_web_routes.py write     # rewrite tests/reference/web_routes.json after an intended change
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

REFERENCE = Path(__file__).with_name("reference") / "web_routes.json"


def snapshot() -> dict:
    from autosteer.web.app import app
    routes = []
    for r in app.router.routes:
        routes.append({"class": type(r).__name__, "path": getattr(r, "path", None),
                       "methods": sorted(getattr(r, "methods", None) or []), "name": getattr(r, "name", None),
                       "include_in_schema": getattr(r, "include_in_schema", None)})
    schema = json.dumps(app.openapi(), sort_keys=True).encode()
    return {"routes": routes, "openapi": hashlib.sha256(schema).hexdigest()[:16],
            "middleware": len(app.user_middleware),
            "exception_handlers": sorted(str(k) for k in app.exception_handlers)}


class WebRoutesTests(unittest.TestCase):
    def test_routes_and_schema_are_unchanged(self):
        want = json.loads(REFERENCE.read_text())
        got = snapshot()
        for key in ("middleware", "exception_handlers", "openapi"):
            self.assertEqual(got[key], want[key], key)
        self.assertEqual([json.dumps(r, sort_keys=True) for r in got["routes"]],
                         [json.dumps(r, sort_keys=True) for r in want["routes"]])


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        REFERENCE.write_text(json.dumps(snapshot(), indent=1) + "\n")
        print("wrote", REFERENCE)
    else:
        unittest.main()
