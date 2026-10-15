import tempfile
import unittest

from fastapi.testclient import TestClient

from autosteer.filler_tools import function_registry

from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.web.app import app


class SeedAuthoringWebTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = DatasetStore(self.tempdir.name)
        app.dependency_overrides[get_dataset_store] = lambda: self.store
        self.client = TestClient(app)
        function_registry.functions["toy_ai_element"] = lambda metadata: {}

    def tearDown(self):
        function_registry.functions.pop("toy_ai_element", None)
        app.dependency_overrides.clear()
        self.tempdir.cleanup()

    def seed_payload(self, **overrides):
        payload = {
            "element_name": "toy_ai_element",
            "question_id": "",
            "sub_id": "0",
            "saved_element_name": "toy_ai_element",
            "question_text": "When income changes from {income_0} to {income_1}, demand changes from {q_0} to {q_1}.",
            "domain": "consumer_theory",
            "type": "comparative_statics",
            "difficulty_level": "2",
            "tags": "",
            "options_text": "",
            "correct_answer": "",
            "instructions": "Write a seed using the same variables.",
            "ai_model": "gpt-5.4-mini",
        }
        payload.update(overrides)
        return payload

    def test_create_edit_update_and_delete_seed(self):
        response = self.client.post("/seeds", data=self.seed_payload())
        self.assertEqual(response.status_code, 200)
        self.assertIn("Seed id 0 saved", response.text)
        self.assertIn("Save changes", response.text)

        dataset = self.store.load_dataset("example_seeds", "toy_ai_element")
        self.assertEqual(len(dataset.example_questions), 1)

        response = self.client.get("/elements/toy_ai_element/example_seeds?dataframe_type=example_questions")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="sc-edit"', response.text)
        self.assertIn('id="sc-delete"', response.text)
        self.assertIn('data-rid="0.0"', response.text)
        self.assertIn("/developer/e/toy_ai_element/seeds/delete-many", response.text)

        response = self.client.get("/seeds/new?element_name=toy_ai_element&question_id=0&sub_id=0")
        self.assertEqual(response.status_code, 200)
        self.assertIn("When income changes", response.text)
        self.assertIn("Instructions for the model", response.text)

        response = self.client.post(
            "/seeds",
            data=self.seed_payload(
                question_id="0",
                question_text="Updated seed with {income} and {quantity}.",
                instructions="Updated instructions.",
            ),
        )
        self.assertEqual(response.status_code, 200)
        dataset = self.store.load_dataset("example_seeds", "toy_ai_element")
        self.assertEqual(len(dataset.example_questions), 1)
        self.assertEqual(dataset.example_questions.iloc[0]["question_text"], "Updated seed with {income} and {quantity}.")
        self.assertEqual(dataset.instruction_set.iloc[0]["instructions"], "Updated instructions.")

        response = self.client.post(
            "/seeds/delete",
            data={"element_name": "toy_ai_element", "question_id": "0", "sub_id": "0"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        dataset = self.store.load_dataset("example_seeds", "toy_ai_element")
        self.assertTrue(dataset.example_questions.empty)
        self.assertTrue(dataset.example_options.empty)
        self.assertTrue(dataset.instruction_set.empty)

    def test_delete_several_seeds(self):
        for text in ("First seed {a}.", "Second seed {b}.", "Third seed {c}."):
            self.assertEqual(self.client.post("/seeds", data=self.seed_payload(question_text=text)).status_code, 200)
        ids = [f"{int(r.question_id)}.{int(r.sub_id)}" for r in self.store.records("example_seeds", "toy_ai_element", dataframe_type="example_questions")]
        self.assertEqual(len(ids), 3)
        response = self.client.post("/developer/e/toy_ai_element/seeds/delete-many",
                                    data={"element_name": "toy_ai_element", "ids": [ids[0], ids[2]]}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        left = self.store.load_dataset("example_seeds", "toy_ai_element").example_questions
        self.assertEqual(list(left["question_text"]), ["Second seed {b}."])

    def test_instruction_parts_build_the_instructions_and_are_kept(self):
        from autosteer import instruction_parts
        parts = dict(parts_form="1", difficulty="1", domain="finance", tags="first_person;rise",
                     part_task="Please give an MCQ example testing the ability to read an income effect",
                     part_type="Income changes and the question asks for the new demand",
                     part_level="", part_domain="household finances",
                     part_tag_keys=["first_person", "rise"],
                     part_tag_texts=["The question should be written in the first person", "Income goes up"])
        preview = self.client.post("/api/seeds/compose", data=self.seed_payload(**parts)).json()
        kinds = [seg["kind"] for seg in preview["segments"]]
        self.assertEqual(kinds[:5], ["task", "domain", "type", "tag", "tag"])
        self.assertIn("household finances", preview["text"])

        response = self.client.post("/seeds", data=self.seed_payload(**parts))
        self.assertEqual(response.status_code, 200)
        saved = self.store.load_dataset("example_seeds", "toy_ai_element").instruction_set.iloc[0]["instructions"]
        self.assertEqual(saved, preview["text"])
        self.assertTrue(saved.startswith("Please give an MCQ example testing the ability to read an income effect. "
                                         "The question should be concerning household finances."))
        kept = instruction_parts.load(self.tempdir.name, "toy_ai_element")
        self.assertEqual(kept["types"]["comparative_statics"], "Income changes and the question asks for the new demand")
        self.assertEqual(kept["domains"]["finance"], "household finances")      # shared: a domain
        self.assertEqual(kept["tags"]["rise"], "Income goes up")                 # the element's own tag
        self.assertNotIn("rise", instruction_parts._read(instruction_parts.shared_file(self.tempdir.name)).get("tags", {}))

        # without a task the seed is not saved, and the editor names the part
        response = self.client.post("/seeds", data=self.seed_payload(**{**parts, "part_task": "",
                                                                        "question_text": "Another {x} seed."}))
        self.assertEqual(response.status_code, 400)
        self.assertIn("write the element part", response.text)

    def test_duplicate_detection_blocks_exact_question_and_warns_on_near_match(self):
        response = self.client.post("/seeds", data=self.seed_payload())
        self.assertEqual(response.status_code, 200)

        response = self.client.post("/seeds", data=self.seed_payload())
        self.assertEqual(response.status_code, 400)
        self.assertIn("matches seed id 0 exactly", response.text)

        near_duplicate = self.seed_payload(
            question_text=(
                "When income changes from {income_0} to {income_1}, "
                "quantity demanded changes from {q_0} to {q_1}."
            )
        )
        response = self.client.post("/seeds", data=near_duplicate)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Likely duplicate of seed id 0", response.text)
        dataset = self.store.load_dataset("example_seeds", "toy_ai_element")
        self.assertEqual(len(dataset.example_questions), 2)
        self.assertEqual(len(dataset.instruction_set), 1)

    def test_missing_example_seed_dataset_renders_empty_state(self):
        self.store.dataset_path("example_seeds", "toy_ai_element").mkdir(parents=True)
        response = self.client.get("/elements/toy_ai_element/example_seeds?dataframe_type=example_questions")
        self.assertEqual(response.status_code, 200)
        self.assertIn("No example seeds yet", response.text)


if __name__ == "__main__":
    unittest.main()
