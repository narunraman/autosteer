import importlib
import json
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from autosteer.filler_tools import function_registry
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.services.generated_seed_service import (
    append_generated_display_seed,
    set_review_status,
)
from autosteer.services.template_generation_service import (
    build_template_targets,
    elements_with_accepted_generated_seeds,
)

app_module = importlib.import_module("autosteer.web.app")


class TemplateGenerationWebTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = DatasetStore(self.tempdir.name)
        self.old_income_filler = function_registry.functions.get("toy_ai_element")
        function_registry.functions["toy_ai_element"] = lambda metadata: {}
        app_module.app.dependency_overrides[get_dataset_store] = lambda: self.store
        self.client = TestClient(app_module.app)

        append_generated_display_seed(
            store=self.store,
            element_name="toy_ai_element",
            display_json={
                "question_1": "Income changes from {income_0} to {income_1}; demand changes from {q_0} to {q_1}.",
                "options_1": [],
                "answer_1": "",
            },
            target_metadata={
                "domain": "finance",
                "type": "comparative_statics",
                "difficulty_level": 2,
                "tags": "first_person",
            },
            instructions="Write a seed testing income elasticity.",
            has_options=False,
        )
        set_review_status(
            self.store,
            "toy_ai_element",
            question_id=0,
            sub_id=0,
            status="accepted",
        )

    def tearDown(self):
        if self.old_income_filler is None:
            function_registry.functions.pop("toy_ai_element", None)
        else:
            function_registry.functions["toy_ai_element"] = self.old_income_filler
        app_module.app.dependency_overrides.clear()
        self.tempdir.cleanup()

    def test_targets_expand_accepted_generated_seeds_across_perspectives(self):
        self.assertEqual(elements_with_accepted_generated_seeds(self.store), ["toy_ai_element"])
        targets = build_template_targets(self.store, "toy_ai_element")
        self.assertEqual(len(targets), 5)
        self.assertEqual({target["domain"] for target in targets}, {"finance"})
        self.assertEqual({target["type"] for target in targets}, {"comparative_statics"})

    def test_template_generation_page_appends_templates_and_ignores_options(self):
        class FakeClient:
            def __init__(self, provider_name):
                self.provider_name = provider_name

            def make_request(self, **kwargs):
                return json.dumps(
                    {
                        "question_1": "A household sees income change from {income_0} to {income_1}.",
                        "options_1": ["The model should ignore this option"],
                        "answer_1": "The model should ignore this option",
                    }
                )

        fake_status = {
            "mode": "developer",
            "ai_configured": True,
            "provider": "OpenAI",
            "openai_configured": True,
            "azure_configured": False,
            "azure_missing_endpoint": False,
            "browser_key_allowed": False,
            "default_model": "gpt-5.4-mini",
            "models": ["gpt-5.4-mini"],
        }

        response = self.client.get("/developer/e/toy_ai_element/templates")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Templates", response.text)
        self.assertIn('id="tgen-button"', response.text)

        with patch.object(app_module, "_developer_status", return_value=fake_status), patch(
            "autosteer.services.template_generation_service.LanguageModelClient",
            FakeClient,
        ):
            response = self.client.post(
                "/templates/generate",
                data={
                    "element_name": "toy_ai_element",
                    "ai_model": "gpt-5.4-mini",
                    "templates_per_target": "2",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Generated 10 templates", response.text)
        dataset = self.store.load_dataset("templates", "toy_ai_element")
        self.assertEqual(len(dataset.questions), 10)
        self.assertTrue(dataset.options.empty)

        response = self.client.get("/elements/toy_ai_element/templates?dataframe_type=questions", follow_redirects=False)
        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/developer/e/toy_ai_element/templates")

    def test_developer_console_no_longer_mentions_step_four(self):
        response = self.client.get("/developer")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Validate and export", response.text)
        self.assertNotIn("Step 4", response.text)
        self.assertIn("/developer/setup", response.text)

    def test_public_mode_sends_authoring_pages_to_the_guide(self):
        with patch.dict("os.environ", {"AUTOSTEER_WEB_MODE": "public"}, clear=False):
            for path in ("/developer", "/seeds/new"):
                response = self.client.get(path, follow_redirects=False)
                self.assertEqual(response.status_code, 303)
                self.assertEqual(response.headers["location"], "/autosteer")

            response = self.client.get("/autosteer")
            self.assertEqual(response.status_code, 200)
            self.assertIn("git clone", response.text)
            self.assertNotIn('href="/developer"', response.text)


if __name__ == "__main__":
    unittest.main()
