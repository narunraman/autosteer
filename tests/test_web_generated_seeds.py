import tempfile
import unittest
import importlib
from unittest.mock import patch

from fastapi.testclient import TestClient

from autosteer.schemas import Metadata, OptionRecord
from autosteer.filler_tools import function_registry
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.services.generated_seed_service import (
    append_generated_display_seed,
    build_generation_prompt,
    build_target_grid,
    generated_seed_records,
    load_review_state,
    set_review_status,
    update_generated_seed,
)
from autosteer.services import generated_seed_validation
app_module = importlib.import_module("autosteer.web.app")
common_module = importlib.import_module("autosteer.web.common")   # what the routes share; patched here


class GeneratedSeedWebTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = DatasetStore(self.tempdir.name)
        self.old_income_filler = function_registry.functions.get("toy_ai_element")
        function_registry.functions["toy_ai_element"] = lambda metadata: {}
        app_module.app.dependency_overrides[get_dataset_store] = lambda: self.store
        self.client = TestClient(app_module.app)
        self.store.save_seed(
            element_name="toy_ai_element",
            question_text="When income changes from {income_0} to {income_1}, demand changes from {q_0} to {q_1}.",
            metadata=Metadata(
                domain="consumer_goods",
                type="comparative_statics",
                difficulty_level=2,
                tags="first_person",
            ),
            options=[OptionRecord(option_id=0, option_text="", correct_answer=True)],
            instructions="Write a seed testing income elasticity.",
        )

    def tearDown(self):
        if self.old_income_filler is None:
            function_registry.functions.pop("toy_ai_element", None)
        else:
            function_registry.functions["toy_ai_element"] = self.old_income_filler
        app_module.app.dependency_overrides.clear()
        self.tempdir.cleanup()

    def test_target_grid_and_prompt_rewrite_target_domain(self):
        targets = build_target_grid(
            self.store,
            "toy_ai_element",
            domains=["finance", "medical"],
        )
        self.assertEqual(len(targets), 2)
        self.assertEqual({target["domain"] for target in targets}, {"finance", "medical"})
        self.assertEqual({target["tags"] for target in targets}, {"first_person"})

        prompt, instructions, seed_templates = build_generation_prompt(self.store, "toy_ai_element", targets[0])
        self.assertIn("finance", prompt)
        self.assertIn("Write a seed testing income elasticity.", prompt)
        self.assertIn("financial settings", instructions)
        self.assertEqual(len(seed_templates), 1)

    def test_generated_seed_page_generates_reviews_and_deletes(self):
        def fake_generate_generated_seed(store, element_name, target_metadata, model, provider):
            prompt, instructions, _ = build_generation_prompt(store, element_name, target_metadata)
            return append_generated_display_seed(
                store=store,
                element_name=element_name,
                display_json={
                    "question_1": "In a {domain_market}, how does demand respond when income changes from {income_0} to {income_1}?",
                    "options_1": [""],
                    "answer_1": "",
                },
                target_metadata=target_metadata,
                instructions=instructions,
                model=model,
                prompt=prompt,
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

        response = self.client.get("/developer/e/toy_ai_element/candidates")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Seed candidates", response.text)
        self.assertIn('id="generate-button"', response.text)

        with patch.object(common_module, "_developer_status", return_value=fake_status), patch.object(
            common_module,
            "generate_generated_seed",
            side_effect=fake_generate_generated_seed,
        ):
            response = self.client.post(
                "/generated-seeds/generate",
                data={
                    "element_name": "toy_ai_element",
                    "domains": ["finance", "medical"],
                    "ai_model": "gpt-5.4-mini",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Generated 2 seeds", response.text)

        records = generated_seed_records(self.store, "toy_ai_element")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["review"]["status"], "draft")
        review = load_review_state(self.store, "toy_ai_element")
        self.assertEqual(len(review), 2)
        self.assertEqual(review["0:0"]["model"], "gpt-5.4-mini")

        response = self.client.post(
            "/generated-seeds/review",
            data={
                "element_name": "toy_ai_element",
                "question_id": "0",
                "sub_id": "0",
                "status": "accepted",
                "notes": "Looks good.",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(load_review_state(self.store, "toy_ai_element")["0:0"]["status"], "accepted")

        response = self.client.post(
            "/generated-seeds/delete",
            data={"element_name": "toy_ai_element", "question_id": "0", "sub_id": "0"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(len(generated_seed_records(self.store, "toy_ai_element")), 1)
        self.assertNotIn("0:0", load_review_state(self.store, "toy_ai_element"))

    def test_ai_validation_fills_accepted_seed_and_saves_result(self):
        append_generated_display_seed(
            store=self.store,
            element_name="toy_ai_element",
            display_json={
                "question_1": "Income rises from {income_0} to {income_1}; demand rises from {q_0} to {q_1}.",
                "options_1": [""],
                "answer_1": "",
            },
            target_metadata={
                "domain": "finance",
                "type": "comparative_statics",
                "difficulty_level": 2,
                "tags": "first_person",
            },
            instructions="Write a seed testing income elasticity.",
        )
        set_review_status(
            self.store,
            "toy_ai_element",
            question_id=0,
            sub_id=0,
            status="accepted",
        )

        function_registry.functions["toy_ai_element"] = lambda metadata: {
            "income_0": 100,
            "income_1": 120,
            "q_0": 10,
            "q_1": 14,
            "options_list": [["elastic", "inelastic"]],
            "correct_answer": [[True, False]],
        }

        with patch.object(generated_seed_validation, "call_solver_model", return_value='{"answers":["A"]}'):
            summary = generated_seed_validation.run_generated_seed_validation(
                store=self.store,
                element_name="toy_ai_element",
                model="gpt-5.4-mini",
                provider="OPENAI",
                runs_per_seed=2,
            )

        self.assertEqual(summary["validated"], 1)
        self.assertEqual(summary["passed"], 1)
        review = load_review_state(self.store, "toy_ai_element")
        solve_check = review["0:0"]["checks"]["solve_check"]
        self.assertEqual(solve_check["status"], "passed")
        self.assertEqual(solve_check["passed_runs"], 2)

    def test_edit_generated_seed_updates_text_and_clears_solve_check(self):
        append_generated_display_seed(
            store=self.store,
            element_name="toy_ai_element",
            display_json={
                "question_1": "Income changes from {income_0} to {income_1}.",
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
        review = load_review_state(self.store, "toy_ai_element")
        review["0:0"]["checks"]["solve_check"] = {"status": "failed", "runs": []}
        from autosteer.services.generated_seed_service import save_review_state
        save_review_state(self.store, "toy_ai_element", review)

        update_generated_seed(
            self.store,
            "toy_ai_element",
            question_id=0,
            sub_id=0,
            question_text="Edited income changes from {income_0} to {income_1}.",
            options_text="Should be ignored",
            correct_answer="0",
        )
        records = generated_seed_records(self.store, "toy_ai_element")
        self.assertEqual(records[0]["question_text"], "Edited income changes from {income_0} to {income_1}.")
        self.assertEqual(records[0]["options"], [])
        review = load_review_state(self.store, "toy_ai_element")
        self.assertNotIn("solve_check", review["0:0"]["checks"])
        self.assertIn("solve_check_history", review["0:0"]["checks"])

    def test_single_generated_seed_validation_checks_one_record(self):
        for domain in ("finance", "medical"):
            append_generated_display_seed(
                store=self.store,
                element_name="toy_ai_element",
                display_json={
                    "question_1": f"{domain} income changes from {{income_0}} to {{income_1}}.",
                    "options_1": [""],
                    "answer_1": "",
                },
                target_metadata={
                    "domain": domain,
                    "type": "comparative_statics",
                    "difficulty_level": 2,
                    "tags": "first_person",
                },
                instructions="Write a seed testing income elasticity.",
            )
        function_registry.functions["toy_ai_element"] = lambda metadata: {
            "income_0": 100,
            "income_1": 120,
            "q_0": 10,
            "q_1": 14,
            "options_list": [["elastic", "inelastic"]],
            "correct_answer": [[True, False]],
        }

        with patch.object(generated_seed_validation, "call_solver_model", return_value='{"answers":["A"]}'):
            result = generated_seed_validation.run_single_generated_seed_validation(
                store=self.store,
                element_name="toy_ai_element",
                question_id=1,
                sub_id=0,
                model="gpt-5.4-mini",
                provider="OPENAI",
                runs_per_seed=1,
            )

        self.assertEqual(result["status"], "passed")
        review = load_review_state(self.store, "toy_ai_element")
        self.assertNotIn("solve_check", review["0:0"]["checks"])
        self.assertEqual(review["1:0"]["checks"]["solve_check"]["status"], "passed")


    def test_background_generation_and_bulk_review(self):
        import time

        def fake_generate_generated_seed(store, element_name, target_metadata, model, provider):
            prompt, instructions, _ = build_generation_prompt(store, element_name, target_metadata)
            return append_generated_display_seed(
                store=store, element_name=element_name,
                display_json={"question_1": "When income goes from {income_0} to {income_1}, what happens to demand?"},
                target_metadata=target_metadata, instructions=instructions, model=model, prompt=prompt)

        fake_status = {"mode": "developer", "ai_configured": True, "provider": "OpenAI", "openai_configured": True,
                       "azure_configured": False, "azure_missing_endpoint": False, "browser_key_allowed": False,
                       "default_model": "gpt-5.4-mini", "models": ["gpt-5.4-mini"]}
        with patch.object(common_module, "_developer_status", return_value=fake_status), patch.object(
                common_module, "generate_generated_seed", side_effect=fake_generate_generated_seed):
            response = self.client.post("/developer/e/toy_ai_element/candidates/generate", follow_redirects=False, data={
                "element_name": "toy_ai_element", "domains": ["finance", "medical", "sports"],
                "ai_model": "gpt-5.4-mini", "background": "1"})
            self.assertEqual(response.status_code, 303)  # back to the page at once; the job runs in a thread
            for _ in range(100):
                job = self.client.get("/developer/e/toy_ai_element/candidates/progress").json()
                if not job["running"]:
                    break
                time.sleep(0.05)
        self.assertEqual((job["total"], job["done"], job["failed"]), (3, 3, 0), job["errors"])
        self.assertEqual(len(generated_seed_records(self.store, "toy_ai_element")), 3)

        def bulk(action, ids):
            r = self.client.post("/developer/e/toy_ai_element/candidates/bulk", follow_redirects=False,
                                 data={"element_name": "toy_ai_element", "action": action, "ids": ids})
            self.assertEqual(r.status_code, 303)

        bulk("accepted", ["0.0", "1.0"])
        bulk("draft", ["1.0"])  # back to needs review
        bulk("delete", ["2.0"])
        review = load_review_state(self.store, "toy_ai_element")
        self.assertEqual(review["0:0"]["status"], "accepted")
        self.assertEqual(review["1:0"]["status"], "draft")
        self.assertEqual(sorted(r["question_id"] for r in generated_seed_records(self.store, "toy_ai_element")), [0, 1])
        page = self.client.get("/developer/e/toy_ai_element/candidates").text
        self.assertIn('id="cand-all"', page)
        self.assertIn('data-act="delete"', page)


if __name__ == "__main__":
    unittest.main()
