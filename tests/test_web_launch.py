"""The public launch: the Evals and datasets switches, pages the public site does not serve, the 404 page,
robots.txt and the sitemap, page metadata, and the limits on /api/sample."""
import importlib
import itertools
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.web import public, site

app_module = importlib.import_module("autosteer.web.app")

PUBLIC = {"AUTOSTEER_WEB_MODE": "public"}
LOCAL = {"AUTOSTEER_WEB_MODE": "local"}
HTML = {"Accept": "text/html,application/xhtml+xml"}
_clients = itertools.count(1)


def env(**values):
    return patch.dict("os.environ", values, clear=False)


def without(*names):
    """The environment with these variables removed for the duration."""
    return patch.dict("os.environ", {k: v for k, v in os.environ.items() if k not in names}, clear=True)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app_module.app.dependency_overrides[get_dataset_store] = lambda: DatasetStore(self.tempdir.name)
        # each test is its own client address, so the /api/sample rate limit does not carry over between tests
        self.client = TestClient(app_module.app, client=(f"10.0.0.{next(_clients)}", 50000))

    def tearDown(self):
        app_module.app.dependency_overrides.clear()
        self.tempdir.cleanup()

    # Evals

    def test_evals_hidden_by_default_on_the_public_site(self):
        with without("AUTOSTEER_EVALS"), env(**PUBLIC):
            self.assertFalse(public.evals_enabled(True))
            for path in ("/", "/scoring", "/tutorial", "/elements/law_of_demand", "/steer-me/setting/2"):
                with self.subTest(path=path):
                    self.assertNotIn('href="/evals', self.client.get(path).text)
            for path in ("/evals", "/evals/steer-me/consumption-decisions-in-non-strategic-environments",
                         "/static/evals.js", "/static/evals_results.json"):
                with self.subTest(path=path):
                    self.assertEqual(self.client.get(path, follow_redirects=False).status_code, 404)
            self.assertNotIn("/evals", self.client.get("/sitemap.xml").text)
            self.assertNotIn("Evals", " ".join(i["label"] for i in site._search_items()))

    def test_evals_can_be_shown_on_the_public_site(self):
        with env(AUTOSTEER_EVALS="on", **PUBLIC):
            self.assertIn('href="/evals"', self.client.get("/").text)
            self.assertIn('href="/evals"', self.client.get("/scoring").text)
            self.assertEqual(self.client.get("/evals").status_code, 200)
            self.assertEqual(self.client.get("/static/evals_results.json").status_code, 200)
            self.assertIn("/evals", self.client.get("/sitemap.xml").text)

    def test_evals_shown_by_default_in_local_mode_and_can_be_hidden(self):
        with without("AUTOSTEER_EVALS"), env(**LOCAL):
            self.assertIn('href="/evals"', self.client.get("/").text)
            self.assertEqual(self.client.get("/evals").status_code, 200)
        with env(AUTOSTEER_EVALS="off", **LOCAL):
            self.assertNotIn('href="/evals"', self.client.get("/").text)
            self.assertEqual(self.client.get("/evals").status_code, 404)

    # Datasets

    def test_dataset_snippets(self):
        with env(**PUBLIC):
            for path in ("/datasets", "/tutorial", "/elements/law_of_demand"):
                with self.subTest(path=path):
                    text = self.client.get(path).text
                    self.assertIn("load_dataset", text)
                    self.assertNotIn("Coming soon.", text)
            element = self.client.get("/elements/law_of_demand").text
            self.assertIn(f'load_dataset("{site.HF_DATASETS["STEER-ME"]}", "law_of_demand")', element)
            self.assertIn('id="tab-ds"', element)
            self.assertIn("ic-hf", self.client.get("/steer").text)

    # What the public site does not serve

    def test_developer_views_are_not_on_the_public_site(self):
        paths = ("/elements/law_of_demand/templates", "/elements/law_of_demand/example_seeds",
                 "/api/datasets/templates/law_of_demand/records", "/docs", "/redoc", "/openapi.json",
                 "/developer/review")
        with env(**PUBLIC):
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(self.client.get(path).status_code, 404)
            self.assertEqual(self.client.get("/elements/law_of_demand").status_code, 200)
            self.assertEqual(self.client.get("/api/elements").status_code, 200)
        with env(**LOCAL):
            self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_held_elements_are_not_on_the_public_site(self):
        held = [e["element"] for e in site.ELEMENTS if e.get("status") == "held"]
        self.assertTrue(held)
        with env(**PUBLIC):
            for name in held:
                with self.subTest(element=name):
                    self.assertEqual(self.client.get(f"/elements/{name}").status_code, 404)
                    self.assertNotIn(f"/elements/{name}<", self.client.get("/sitemap.xml").text)
        with env(**LOCAL):
            self.assertEqual(self.client.get(f"/elements/{held[0]}").status_code, 200)

    def test_facet_links_are_url_encoded(self):
        entries = {"law_of_demand": {"types": ["compute intersect", "slope"], "domains": ["finance"], "tags": ["anon"]}}
        with env(**PUBLIC), patch.object(site, "load_manifest", return_value={"elements": entries}):
            page = self.client.get("/elements/law_of_demand").text
            self.assertIn('href="/by/type/compute%20intersect"', page)
            self.assertEqual(self.client.get("/by/type/compute%20intersect").status_code, 200)

    def test_not_found_page(self):
        with env(**PUBLIC):
            page = self.client.get("/no/such/page", headers=HTML)
            self.assertEqual(page.status_code, 404)
            self.assertIn("Page not found", page.text)
            self.assertIn('name="robots" content="noindex"', page.text)
            self.assertIn('href="/tutorial"', page.text)
            self.assertEqual(self.client.get("/elements/no_such_element", headers=HTML).status_code, 404)
            self.assertIn("Page not found", self.client.get("/elements/no_such_element", headers=HTML).text)
            # API clients and static files keep a JSON answer
            api = self.client.get("/api/elements/no_such_element", headers=HTML)
            self.assertEqual(api.status_code, 404)
            self.assertIn("detail", api.json())
            self.assertEqual(self.client.get("/nope").json(), {"detail": "Not Found"})

    # Robots, sitemap, metadata

    def test_robots_and_sitemap(self):
        with env(**PUBLIC):
            robots = self.client.get("/robots.txt").text
            self.assertIn("Allow: /", robots)
            self.assertIn("Sitemap: https://steer-benchmark.cs.ubc.ca/sitemap.xml", robots)
            xml = self.client.get("/sitemap.xml")
            self.assertEqual(xml.headers["content-type"].split(";")[0], "application/xml")
            locs = re.findall(r"<loc>https://steer-benchmark\.cs\.ubc\.ca(.*?)</loc>", xml.text)
            self.assertIn("/", locs)
            self.assertIn("/reference", locs)
            self.assertIn("/elements/law_of_demand", locs)
            self.assertEqual(len(locs), len(set(locs)))
            for path in locs[:12] + locs[-5:]:
                with self.subTest(path=path):
                    self.assertEqual(self.client.get(path).status_code, 200)
            self.assertEqual(self.client.get("/favicon.ico", follow_redirects=False).headers["location"],
                             "/static/favicon.svg")
            for name in ("favicon.svg", "og.png", "apple-touch-icon.png"):
                self.assertEqual(self.client.get(f"/static/{name}").status_code, 200)

    def test_titles_descriptions_and_previews(self):
        def head(path):
            text = self.client.get(path).text
            title = re.search(r"<title>(.*?)</title>", text).group(1)
            meta = dict(re.findall(r'<meta (?:name|property)="([^"]+)" content="([^"]*)"', text))
            return title, meta, text

        with env(**PUBLIC):
            titles = set()
            for path in ("/", "/tutorial", "/benchmarks", "/scoring", "/reference",
                         "/steer-me/setting/2", "/steer/module/2.1", "/elements/law_of_demand", "/by/domain/finance"):
                with self.subTest(path=path):
                    title, meta, text = head(path)
                    titles.add(title)
                    # the tab shows just the page's name; link previews carry the site's name too
                    self.assertTrue(title)
                    self.assertTrue(meta.get("description"))
                    self.assertIn("STEER", meta["og:title"])
                    self.assertIn(title, meta["og:title"])
                    self.assertEqual(meta["twitter:title"], meta["og:title"])
                    self.assertEqual(meta["og:url"], "https://steer-benchmark.cs.ubc.ca" + path)
                    self.assertEqual(meta["og:image"], "https://steer-benchmark.cs.ubc.ca/static/og.png")
                    self.assertEqual(meta["twitter:card"], "summary_large_image")
                    self.assertIn(f'<link rel="canonical" href="https://steer-benchmark.cs.ubc.ca{path}">', text)
            self.assertEqual(len(titles), 9)
            self.assertEqual(head("/elements/law_of_demand")[0], "Law of Demand")
            self.assertEqual(head("/")[0], "STEER")
        with env(AUTOSTEER_SITE_URL="http://localhost:9000/", **PUBLIC):
            self.assertEqual(head("/reference")[1]["og:url"], "http://localhost:9000/reference")

    # /api/sample limits

    def test_seeded_samples_are_cached_and_random_ones_are_not(self):
        with env(**PUBLIC):
            params = {"element_name": "add_sub", "n": 2, "seed": 1234}
            first = self.client.get("/api/sample", params=params)
            again = self.client.get("/api/sample", params=dict(reversed(list(params.items()))))
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json(), again.json())
            self.assertEqual(again.headers["x-cache"], "hit")
            self.assertIn("max-age", first.headers["cache-control"])
            fresh = self.client.get("/api/sample", params={"element_name": "add_sub"})
            self.assertEqual(fresh.status_code, 200)
            self.assertEqual(fresh.headers["cache-control"], "no-store")
            self.assertEqual(self.client.get("/api/sample", params={"element_name": "nope", "seed": 1}).headers
                             ["cache-control"], "no-store")

    def test_rate_limit(self):
        with env(AUTOSTEER_API_RATE="3", **PUBLIC):
            codes = [self.client.get("/api/sample", params={"element_name": "add_sub", "seed": 5}).status_code
                     for _ in range(5)]
            self.assertEqual(codes, [200, 200, 200, 429, 429])
            limited = self.client.get("/api/sample", params={"element_name": "add_sub", "seed": 5})
            self.assertGreater(int(limited.headers["retry-after"]), 0)
            # other pages are not limited
            self.assertEqual(self.client.get("/reference").status_code, 200)
        with env(AUTOSTEER_API_RATE="0", **PUBLIC):
            self.assertEqual(self.client.get("/api/sample", params={"element_name": "add_sub", "seed": 5}).status_code, 200)
        with without("AUTOSTEER_API_RATE"):
            self.assertEqual(public.api_rate(True), 60)
            self.assertEqual(public.api_rate(False), 0)

    def test_request_size_limits(self):
        with env(**PUBLIC):
            self.assertEqual(self.client.get("/api/sample?element_name=" + "a" * 3000).status_code, 414)
            big = self.client.post("/api/sample", content=b'{"element_name": "' + b"a" * 20000 + b'"}',
                                   headers={"content-type": "application/json"})
            self.assertEqual(big.status_code, 413)
            ok = self.client.post("/api/sample", json={"element_name": "add_sub", "n": 1, "seed": 3})
            self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
