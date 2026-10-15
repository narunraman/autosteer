import importlib
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.web import site

app_module = importlib.import_module("autosteer.web.app")


class SiteWebTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app_module.app.dependency_overrides[get_dataset_store] = lambda: DatasetStore(self.tempdir.name)
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.app.dependency_overrides.clear()
        self.tempdir.cleanup()

    def test_guide_pages_render(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("@inproceedings", response.text)

        paths = [item["href"] for group in site.GUIDE_NAV for item in group["items"]] + ["/reference"]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

        self.assertEqual(self.client.get("/no-such-page").status_code, 404)

    def test_taxonomy_pages_render_for_both_benchmarks(self):
        for version, vslug in site.SLUGS.items():
            for setting in site.tree(version):
                with self.subTest(benchmark=vslug, setting=setting["number"]):
                    response = self.client.get(f"/{vslug}/setting/{setting['number']}")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(setting["modules"][0]["name"], response.text)
            module = site.tree(version)[1]["modules"][0]
            response = self.client.get(f"/{vslug}/module/{module['number']}")
            self.assertEqual(response.status_code, 200)

        self.assertEqual(self.client.get("/steer/setting/99").status_code, 404)
        self.assertEqual(self.client.get("/steer/module/9.9").status_code, 404)
        self.assertEqual(self.client.get("/other/setting/1").status_code, 404)

    def test_every_described_element_has_a_page(self):
        for element in site.ELEMENTS:
            with self.subTest(element=element["element"]):
                response = self.client.get(f"/elements/{element['element']}")
                self.assertEqual(response.status_code, 200)

        self.assertEqual(self.client.get("/elements/no_such_element").status_code, 404)

    def test_elements_can_be_listed_by_domain_type_or_perspective(self):
        entries = {
            "law_of_demand": {"types": ["price_increase"], "domains": ["finance", "sports"], "tags": ["anon", "woman"]},
            "consumer_surplus": {"types": ["linear"], "domains": ["finance"], "tags": ["anon"]},
        }
        with patch.object(site, "load_manifest", return_value={"elements": entries}):
            response = self.client.get("/by/domain/finance")
            self.assertEqual(response.status_code, 200)
            self.assertIn('class="row-title" href="/elements/law_of_demand"', response.text)
            self.assertIn('class="row-title" href="/elements/consumer_surplus"', response.text)

            response = self.client.get("/by/domain/sports")
            self.assertIn('class="row-title" href="/elements/law_of_demand"', response.text)
            self.assertNotIn('class="row-title" href="/elements/consumer_surplus"', response.text)

            self.assertEqual(self.client.get("/by/perspective/woman").status_code, 200)
            self.assertEqual(self.client.get("/by/type/linear").status_code, 200)
            self.assertEqual(self.client.get("/by/domain/nowhere").status_code, 404)
            self.assertEqual(self.client.get("/by/colour/red").status_code, 404)

            self.assertIn('href="/by/domain/sports"', self.client.get("/elements/law_of_demand").text)

    def test_evals_pages_render(self):
        for path in ("/evals", "/evals/steer-me/consumption-decisions-in-non-strategic-environments", "/evals/steer-me/consumption-decisions-in-non-strategic-environments/labor-supply"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn('id="evals-data"', response.text)
        self.assertEqual(self.client.get("/evals/nothing").status_code, 404)
        self.assertEqual(self.client.get("/evals/steer-me/nothing").status_code, 404)
        data = site.evals_data()
        self.assertEqual({b["key"] for b in data["benchmarks"]}, {"STEER", "STEER-ME"})

    def test_published_evals_results_hold_back_incomplete_runs(self):
        results = site.eval_results()
        for i, m in enumerate(results["models"]):
            for version in site.SLUGS:
                has = any(str(i) in cells for cells in results["benchmarks"].get(version, {}).values())
                with self.subTest(model=m["name"], benchmark=version):
                    self.assertEqual(has, site.eval_listed(results, i, version))
        self.assertEqual(site.publishable_eval_results(results)["models"], results["models"])

    def test_evals_results_load_and_name_current_elements(self):
        results = site.eval_results()
        self.assertEqual(results["schema"], "autosteer-evals-results/1")
        self.assertEqual(results["metrics"], ["norm", "exact", "ece", "brier", "epa"])
        scored = results["benchmarks"]["STEER-ME"]
        self.assertGreater(len(scored), 20)
        shown = {e["id"] for b in site.evals_data()["benchmarks"] if b["key"] == "STEER-ME"
                 for s in b["settings"] for m in s["modules"] for e in m["elements"]}
        for element, by_model in scored.items():
            with self.subTest(element=element):
                self.assertIn(element, site.ELEMENTS_BY_NAME)
                # STEER elements appear here only as the "Shared with STEER" module (checked below: shown)
                self.assertIn(site.ELEMENTS_BY_NAME[element]["version"], ("STEER-ME", "STEER"))
                self.assertNotIn(element, site.ALIASES)
                self.assertIn(element, shown)
                for index, formats in by_model.items():
                    self.assertLess(int(index), len(results["models"]))
                    for key, vector in formats.items():
                        if key != "by":
                            self.assertIn(len(vector), (6, 7))  # 7th: share of questions with no usable answer
                            self.assertGreater(vector[5], 0)
        # Runs without option probabilities carry no calibration numbers.
        for index, model in enumerate(results["models"]):
            if not model["logprobs"]:
                vectors = [f["mc"] for by_model in scored.values() for i, f in by_model.items() if int(i) == index and "mc" in f]
                self.assertTrue(vectors)
                self.assertTrue(all(v[2] is None and v[3] is None and v[4] is None for v in vectors))

    def test_evals_models_and_formats_come_from_the_results(self):
        data = site.evals_data()
        results = site.eval_results()
        self.assertEqual([m["name"] for m in data["models"]], [m["name"] for m in results["models"]])
        self.assertIn("GPT-4o", [m["name"] for m in data["models"]])
        self.assertFalse(hasattr(site, "EVAL_MODELS"))
        formats = {f["key"]: f["scored"] for f in data["formats"]}
        self.assertTrue(formats["mc"])
        self.assertEqual(formats["free"], "free" in results["formats"])
        self.assertEqual({b["key"]: b["scored"] for b in data["benchmarks"]}["STEER"], bool(results["benchmarks"].get("STEER")))

    def test_evals_page_reads_the_scored_results_not_a_generator(self):
        page = self.client.get("/evals")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn("mock-badge", page.text)
        self.assertIn("Click a setting to go inside it, and click a score to see how it breaks down.", page.text)
        self.assertIn("evals_results.json", page.text)
        static = self.client.get("/static/evals_results.json")
        self.assertEqual(static.status_code, 200)
        self.assertEqual(static.json()["schema"], "autosteer-evals-results/1")
        script = self.client.get("/static/evals.js").text
        self.assertIn("fetch(out.dataset.src)", script)
        for placeholder in ("placeholder numbers", "2166136261", "Math.random", "skill(", "CLOSED"):
            self.assertNotIn(placeholder, script)

    def test_merged_elements_redirect_and_shared_and_covered_elements_show(self):
        response = self.client.get("/elements/comp_stat_cobb", follow_redirects=False)
        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/elements/input_price_response")
        foundations = site.tree("STEER-ME")[0]
        shared = [m for m in foundations["modules"] if m["name"] == "Shared with STEER"]
        self.assertEqual(len(shared), 1)
        self.assertIn("add_sub", [e["element"] for e in shared[0]["elements"]])
        page = self.client.get(f"/steer-me/module/{shared[0]['number']}")
        self.assertEqual(page.status_code, 200)
        self.assertIn('href="/elements/add_sub"', page.text)
        covered = [e for s in site.tree("STEER") for m in s["modules"] for e in m["elements"] if e["status"] == "covered"]
        self.assertEqual({e["id"] for e in covered}, {"2.4.c", "2.4.d"})
        self.assertTrue(all(e["covered_by"] == "maximize_expected_utility" for e in covered))

    def test_developer_link_only_shows_in_local_mode(self):
        self.assertIn('href="/developer"', self.client.get("/").text)
        with patch.dict("os.environ", {"AUTOSTEER_WEB_MODE": "public"}, clear=False):
            self.assertNotIn('href="/developer"', self.client.get("/").text)


if __name__ == "__main__":
    unittest.main()
