"""/llms.txt and the Evals results as plain data (/evals.csv, /evals.json) follow the Evals switch, and the CSV
holds every published number once."""
import csv
import io

import pytest
from fastapi.testclient import TestClient

from autosteer.web import site
from autosteer.web.app import create_app


@pytest.fixture()
def client(monkeypatch):
    def make(evals: bool):
        monkeypatch.setenv("AUTOSTEER_EVALS", "1" if evals else "0")
        return TestClient(create_app())
    return make


def test_llms_txt_lists_the_guide_and_the_evals_data(client):
    text = client(True).get("/llms.txt").text
    assert text.startswith("# STEER\n\n> ")
    for g in site.GUIDE_NAV:
        for i in g["items"]:
            assert f"]({i['href'] if i['href'] != '/' else ''}" in text.replace(site.public.site_url(), "")
    assert "/evals.csv" in text
    assert "/evals.csv" not in client(False).get("/llms.txt").text


def test_evals_data_follows_the_switch(client):
    off = client(False)
    assert off.get("/evals.csv").status_code == 404 and off.get("/evals.json").status_code == 404


def test_evals_csv_has_every_published_number_once(client):
    rows = list(csv.DictReader(io.StringIO(client(True).get("/evals.csv").text)))
    results = site.eval_results()
    top = sum(1 for els in results["benchmarks"].values() for per in els.values() for e in per.values()
              for f, v in e.items() if f not in ("by", "nocot") and isinstance(v, list))
    assert sum(r["slice"] == "all" for r in rows) == top
    keys = [(r["benchmark"], r["element"], r["model"], r["format"], r["slice"], r["slice_value"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert {r["model"] for r in rows} <= {m["name"] for m in results["models"]}
    assert client(True).get("/evals.json").json()["csv"].endswith("/evals.csv")
