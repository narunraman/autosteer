"""Developer mode as an element workspace: routes, redirects from the old stage pages, New element, Setup."""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from autosteer.schemas import Metadata, OptionRecord
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.web import devmode, review, site

app_module = importlib.import_module("autosteer.web.app")
common_module = importlib.import_module("autosteer.web.common")   # what the routes share; patched here
CODE_ELEMENT = "pure_nash"  # on the site, templates written in code


class DevmodeWebTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = DatasetStore(self.tempdir.name)
        app_module.app.dependency_overrides[get_dataset_store] = lambda: self.store
        self.client = TestClient(app_module.app)
        self.review_dir = patch.object(review, "REVIEW_DIR", Path(self.tempdir.name) / "review")
        self.review_dir.start()
        self.env = patch.dict(os.environ, {"AUTOSTEER_WEB_MODE": "local"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.review_dir.stop()
        app_module.app.dependency_overrides.clear()
        self.tempdir.cleanup()

    def get(self, url, **kw):
        return self.client.get(url, follow_redirects=False, **kw)

    def add_seed(self, element):
        self.store.save_seed(
            element_name=element,
            question_text="A price changes from {p_0} to {p_1}.",
            metadata=Metadata(domain="finance", type="linear", difficulty_level=0, tags="first_person"),
            options=[OptionRecord(option_id=0, option_text="", correct_answer=True)],
            instructions="Write a seed.",
        )

    # Pages -------------------------------------------------------------------------------------------------
    def test_dashboard_and_new_pages(self):
        for url, text in (("/developer", "<h1>Dashboard</h1>"), ("/developer/new", "New element")):
            response = self.get(url)
            self.assertEqual(response.status_code, 200, url)
            self.assertIn(text, response.text)
        dash = self.get("/developer").text
        self.assertNotIn('class="dev-crumbs"', dash)  # the title says where you are
        # The review ring heads the filter toolbar and lists the elements not reviewed.
        self.assertIn('class="dash-ring" href="/developer?review=not"', dash)
        self.assertIn('class="dash-tools el-side"', dash)
        self.assertNotIn("Template data in", dash)
        # Sidebar: Workspace, Guide and Settings are labels; their pages are the links.
        for group in ("Workspace", "Guide", "Settings"):
            self.assertIn(f'<div class="group">{group}</div>', dash)
        for href in ("/developer/new", "/developer/how", "/developer/submission", "/developer/setup"):
            self.assertIn(f'href="{href}"', dash)
        self.assertLess(dash.index(">How elements are made</a>"), dash.index(">AI setup</a>"))
        how = self.get("/developer/how").text
        for anchor in ("seeds", "candidates", "templates", "review", "code"):
            self.assertIn(f'id="{anchor}"', how)

    def test_guide_and_settings_pages(self):
        for url, title in (("/developer/how", "How elements are made"), ("/developer/submission", "Review and submission"),
                           ("/developer/setup", "AI setup")):
            page = self.get(url).text
            self.assertIn(f"<h1>{title}</h1>", page)
            self.assertIn(f"<title>{title} · Developer</title>", page)
            self.assertNotIn('class="dev-crumbs"', page)
            self.assertIn(f'href="{url}" aria-current="page"', page)
        how = self.get("/developer/how").text
        self.assertIn("Before you start", how)
        self.assertIn("Never let a model write an answer.", how)
        self.assertIn("You accept, edit or reject every candidate.", how)
        self.assertLess(how.index("Before you start"), how.index("The order we support"))
        self.assertIn("pull request", self.get("/developer/submission").text)
        self.assertEqual(self.get("/developer/status").headers["location"], "/developer")
        self.assertIn("Providers", self.get("/developer/setup").text)

    def test_a_key_entered_on_setup_is_used_and_never_shown(self):
        secret = "sk-test-SESSION-KEY-0123456789"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            response = self.client.post("/developer/setup/key", data={"name": "ANTHROPIC_API_KEY", "value": secret})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(secret, response.text)
            self.assertIn("ANTHROPIC_API_KEY is set until the server stops.", response.text)
            self.assertIn("set for this session", response.text)
            self.assertTrue(common_module._developer_status()["ai_configured"])
            self.assertEqual(self.client.post("/developer/setup/key", data={"name": "PATH", "value": "x"}).status_code, 400)
            response = self.client.post("/developer/setup/forget")
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
            self.assertFalse(common_module._developer_status()["ai_configured"])
        new = self.get("/developer/new").text
        self.assertIn('<a href="/developer">Dashboard</a><span class="sep" aria-hidden="true">›</span><span aria-current="page">New element</span>', new)
        self.assertIn('<a href="/developer/new" aria-current="page">New element</a>', new)
        elements = self.get("/developer").text
        for bench in ("STEER", "STEER-ME", "Drafts"):
            self.assertIn(f'data-bench="{bench if bench != "Drafts" else "draft"}">{bench} <span>', elements)
        self.assertIn(">Pure Nash Equilibrium</a>", elements)
        self.assertIn('<span class="built built-code">Code</span>', elements)
        self.assertIn('class="el-key"', elements)
        for f in ("f-setting", "f-module", "f-built", "f-status", "f-q"):
            self.assertIn(f'id="{f}"', elements)
        # Settings and modules from the taxonomy, with the papers' numbers.
        first = site.tree("STEER")[0]
        self.assertIn(f'<span class="dev-num">{first["number"]}</span> {first["name"]}</th>', elements)

    def test_element_overview_and_steps_name_the_element(self):
        d = site.ELEMENTS_BY_NAME[CODE_ELEMENT]
        response = self.get(f"/developer/e/{CODE_ELEMENT}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(f"<title>{d['display_name']} · Developer</title>", response.text)
        self.assertIn(f"<h1>{d['display_name']}</h1>", response.text)
        self.assertIn("written in code", response.text)
        # Code elements: Example seeds and Seed candidates are not links, and their URLs go to the overview;
        # Templates lists the templates written in code (read only)
        self.assertIn(f'href="/developer/e/{CODE_ELEMENT}/templates"', response.text)
        self.assertEqual(self.get(f"/developer/e/{CODE_ELEMENT}/templates").status_code, 200)
        for key in ("seeds", "candidates"):
            self.assertNotIn(f'href="/developer/e/{CODE_ELEMENT}/{key}"', response.text)
            r = self.get(f"/developer/e/{CODE_ELEMENT}/{key}")
            self.assertEqual((r.status_code, r.headers["location"]), (302, f"/developer/e/{CODE_ELEMENT}"))
        response = self.get(f"/developer/e/{CODE_ELEMENT}/review")
        self.assertEqual(response.status_code, 200)
        self.assertIn(f"<title>4. Review · {d['display_name']} · Developer</title>", response.text)
        self.assertIn(f'href="/developer/e/{CODE_ELEMENT}/review" aria-current="page"', response.text)
        self.assertIn(f'<span aria-current="page">Review</span>', response.text)

    def test_ai_element_breadcrumb_sidebar_and_step_links(self):
        self.add_seed("toy_element")
        page = self.get("/developer/e/toy_element/candidates").text
        # Breadcrumb: Dashboard › element › step, every crumb but the last a link, names as in the sidebar.
        self.assertIn('<nav class="dev-crumbs" aria-label="Breadcrumb"><a href="/developer">Dashboard</a>', page)
        self.assertIn('<a href="/developer/e/toy_element">Toy Element</a><span class="sep" aria-hidden="true">›</span><span aria-current="page">Seed candidates</span>', page)
        self.assertIn("<title>2. Seed candidates · Toy Element · Developer</title>", page)
        # Sidebar: the open element and its four steps sit under Dashboard; Guide and Settings pages stay listed.
        self.assertRegex(page, r'<a href="/developer">Dashboard</a>\s*<ul class="dev-tree-el">')
        self.assertIn('href="/developer/setup"', page)
        self.assertIn('href="/developer/new"', page)
        self.assertIn('href="/developer/e/toy_element/candidates" aria-current="page"', page)
        for key in ("seeds", "templates", "review"):
            self.assertIn(f'href="/developer/e/toy_element/{key}"', page)
        self.assertNotIn("‹ All elements", page)
        self.assertNotIn("autoSTEER guide", page)
        templates = self.get("/developer/e/toy_element/templates").text
        self.assertIn("Go to Seed candidates →", templates)
        overview = self.get("/developer/e/toy_element").text
        self.assertIn('href="/developer/e/toy_element/candidates"', overview)
        for text in (templates, overview, page):
            self.assertNotRegex(text, r"in step \d|step \d,|steps \d")

    def test_external_links_open_in_a_new_tab(self):
        page = self.get(f"/developer/e/{CODE_ELEMENT}").text
        self.assertIn(f'href="/elements/{CODE_ELEMENT}" target="_blank" rel="noopener">Public page ↗</a>', page)
        self.assertIn(f'target="_blank" rel="noopener">Sample a question ↗</a>', page)

    def test_elements_link_to_their_overview(self):
        home = self.get("/developer").text
        self.assertIn(f'<a href="/developer/e/{CODE_ELEMENT}">', home)
        # the name opens the overview; only the step dots link into a step
        self.assertNotRegex(home, r'<td class="el-name"><a href="/developer/e/[a-z0-9_]+/')
        self.assertRegex(home, r'<a class="step-dot" href="/developer/e/[a-z0-9_]+/(seeds|candidates|templates|review)"')
        self.assertNotIn('class="dev-crumbs"', home)
        self.assertIn('class="dash-tools el-side"', home)  # the filters in the toolbar on the right

    def test_template_list_is_paged(self):
        rows = self.get(f"/developer/e/{CODE_ELEMENT}/template-rows?offset=0").json()
        self.assertEqual(len(rows["rows"]), min(25, rows["total"]))
        page = self.get(f"/developer/e/{CODE_ELEMENT}/templates").text
        self.assertEqual(page.count(f'"template_id": "{CODE_ELEMENT}/'), min(25, rows["total"]))
        found = self.get(f"/developer/e/{CODE_ELEMENT}/template-rows?q=equilibrium").json()
        for r in found["rows"]:
            self.assertIn("equilibrium", " ".join(x.get("text", "") for p in r["pieces"] for x in p).lower())

    def test_review_sample_is_fixed_and_follows_the_rule(self):
        from autosteer.web import review
        first, again = review.review_sample(CODE_ELEMENT), review.review_sample(CODE_ELEMENT)
        self.assertEqual([p["key"] for p in first["picks"]], [p["key"] for p in again["picks"]])
        self.assertEqual([p["fill_seed"] for p in first["picks"]], [p["fill_seed"] for p in again["picks"]])
        self.assertEqual(first["n"], review.sample_size(first["total"]))
        self.assertEqual(len(first["picks"]), first["n"])
        self.assertEqual([review.sample_size(t) for t in (3, 10, 400, 600, 5000)], [3, 10, 10, 12, 20])
        page = self.get(f"/developer/e/{CODE_ELEMENT}/review").text
        self.assertIn(f"A sample of <b>{first['n']}</b>", page)
        self.assertNotIn("Draw a new sample", page)

    def test_repeated_text_at_a_level_is_one_template(self):
        from autosteer.web.review import distinct_templates
        row = lambda i, t: {"id": i, "template_id": f"x/questions/{i}", "parts": ["What is {a} over {b}?"], "type": t,
                            "domain": "numbers", "tags": "divide", "difficulty": 1, "difficulty_raw": 0}
        out = distinct_templates([row(0, "positive"), row(1, "negative"), row(2, "both"), {**row(3, "both"), "difficulty": 2}])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["fills"], {"type": ["both", "negative", "positive"]})
        self.assertEqual(len(out[0]["variants"]), 3)

    def test_saving_a_review_comes_back_with_a_message(self):
        response = self.client.post(f"/developer/e/{CODE_ELEMENT}/review", data={"notes": ""})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Notes saved.", response.text)

    def test_unknown_element_and_step(self):
        self.assertEqual(self.get("/developer/e/no_such_element_xyz").status_code, 404)
        self.assertEqual(self.get(f"/developer/e/{CODE_ELEMENT}/nonsense").status_code, 404)

    def test_ai_element_steps_show_its_data(self):
        self.add_seed("toy_element")
        response = self.get("/developer/e/toy_element/seeds")
        self.assertEqual(response.status_code, 200)
        self.assertIn('<span class="slot">{p_0}</span>', response.text)
        self.assertIn('data-rid="0.0"', response.text)  # selected, then Edit opens /seeds?edit=0.0
        editor = self.get("/developer/e/toy_element/seeds?edit=0.0").text
        self.assertIn("Save changes", editor)
        self.assertIn('action="/developer/e/toy_element/seeds"', editor)
        candidates = self.get("/developer/e/toy_element/candidates").text
        self.assertIn("Generate", candidates)
        self.assertIn('No AI provider. <a href="/developer/setup">Setup</a>', candidates)
        self.assertIn("disabled", candidates)
        self.assertIn("1 seed", self.get("/developer/e/toy_element/seeds").text)

    # Redirects ---------------------------------------------------------------------------------------------
    def test_old_urls_redirect_to_the_workspace(self):
        cases = {
            "/seeds/new": "/developer",
            f"/seeds/new?element_name={CODE_ELEMENT}": f"/developer/e/{CODE_ELEMENT}/seeds?edit=new",
            f"/seeds/new?element_name={CODE_ELEMENT}&question_id=3&sub_id=1": f"/developer/e/{CODE_ELEMENT}/seeds?edit=3.1",
            "/generated-seeds": "/developer",
            f"/generated-seeds?element_name={CODE_ELEMENT}": f"/developer/e/{CODE_ELEMENT}/candidates",
            f"/templates/generate?element_name={CODE_ELEMENT}": f"/developer/e/{CODE_ELEMENT}/templates",
            f"/elements/{CODE_ELEMENT}/example_seeds?dataframe_type=example_questions": f"/developer/e/{CODE_ELEMENT}/seeds",
            f"/elements/{CODE_ELEMENT}/generated_seeds": f"/developer/e/{CODE_ELEMENT}/candidates",
            f"/elements/{CODE_ELEMENT}/templates?dataframe_type=questions": f"/developer/e/{CODE_ELEMENT}/templates",
            "/developer/review": "/developer?review=not",
            "/developer?view=review": "/developer?review=not",
            "/developer/elements?status=held": "/developer?status=held",
            f"/developer/review/{CODE_ELEMENT}": f"/developer/e/{CODE_ELEMENT}/review",
        }
        for old, new in cases.items():
            response = self.get(old)
            self.assertEqual(response.status_code, 301, old)
            self.assertEqual(response.headers["location"], new, old)

    def test_merged_element_ids_redirect(self):
        # An old id with no data folder left goes to the element it was merged into; one that still has a folder
        # stays viewable as a legacy folder that says where it went.
        known = devmode.known_elements(self.store)
        gone = [a for a in site.ALIASES if a not in known]
        kept = [a for a in site.ALIASES if a in known]
        if gone:
            response = self.get(f"/developer/e/{gone[0]}/review")
            self.assertEqual(response.status_code, 301)
            self.assertEqual(response.headers["location"], f"/developer/e/{site.ALIASES[gone[0]]}/review")
        if kept:
            response = self.get(f"/developer/e/{kept[0]}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("merged into", response.text)
        if not site.ALIASES:
            self.skipTest("no element aliases")

    def test_public_mode_hides_developer_pages(self):
        with patch.dict(os.environ, {"AUTOSTEER_WEB_MODE": "public"}):
            for url in ("/developer", "/developer/elements", "/developer/new", "/developer/setup", f"/developer/e/{CODE_ELEMENT}",
                        f"/developer/e/{CODE_ELEMENT}/candidates", "/generated-seeds"):
                response = self.get(url)
                self.assertEqual(response.status_code, 303, url)
                self.assertEqual(response.headers["location"], "/autosteer")
            for url in (f"/elements/{CODE_ELEMENT}/templates", "/developer/review", f"/developer/review/{CODE_ELEMENT}"):
                self.assertEqual(self.get(url).status_code, 404, url)

    # New element -------------------------------------------------------------------------------------------
    def new_element_form(self, **overrides):
        setting = site.tree("STEER-ME")[1]
        form = {"display_name": "Toy Surplus Element", "element_id": "", "version": "STEER-ME",
                "setting": setting["name"], "module": setting["modules"][0]["name"], "summary": "Computing a toy surplus."}
        form.update(overrides)
        return form

    def test_new_element_creates_a_draft_and_opens_step_one(self):
        response = self.client.post("/developer/new", data=self.new_element_form(), follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers["location"].startswith("/developer/e/toy_surplus_element/seeds"))
        draft = json.loads((Path(self.tempdir.name) / "devmode" / "elements" / "toy_surplus_element.json").read_text())
        self.assertEqual(draft["status"], "draft")
        self.assertEqual(draft["display_name"], "Toy Surplus Element")
        self.assertTrue(self.store.dataset_path("example_seeds", "toy_surplus_element").is_dir())
        page = self.client.get(response.headers["location"]).text
        self.assertIn("Created Toy Surplus Element", page)
        self.assertIn("Toy Surplus Element", self.get("/developer").text)
        self.assertNotIn("toy_surplus_element", json.dumps(site.ELEMENTS))
        # A seed can be saved for the draft even though no filler is registered under its id yet.
        response = self.client.post("/developer/e/toy_surplus_element/seeds", data={
            "element_name": "toy_surplus_element", "question_id": "", "sub_id": "0", "saved_element_name": "toy_surplus_element",
            "question_text": "Demand is {a} - {b}p.", "domain": "finance", "type": "linear", "difficulty_level": "0",
            "tags": "", "options_text": "", "correct_answer": "", "instructions": "Write a seed.", "ai_model": ""})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Seed id 0 saved", response.text)

    def test_new_element_rejects_a_taken_id(self):
        response = self.client.post("/developer/new", data=self.new_element_form(display_name="Pure Nash"), follow_redirects=False)
        self.assertEqual(response.status_code, 400)
        self.assertIn("already used", response.text)
        response = self.client.post("/developer/new", data=self.new_element_form(module="Not a module"), follow_redirects=False)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose a module", response.text)

    def test_new_element_with_a_new_setting_and_module(self):
        form = self.new_element_form(display_name="Ambiguity Toy", setting="__new__", module="__new__",
                                     new_setting="Decisions under Ambiguity", new_module="Ellsberg Choices")
        response = self.client.post("/developer/new", data=form, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        draft = json.loads((Path(self.tempdir.name) / "devmode" / "elements" / "ambiguity_toy.json").read_text())
        self.assertEqual((draft["setting"], draft["module"]), ("Decisions under Ambiguity", "Ellsberg Choices"))
        self.assertTrue(draft["custom_setting"] and draft["custom_module"])
        self.assertIn("Ellsberg Choices", self.get("/developer/e/ambiguity_toy").text)
        # A later draft is offered the new setting and module, and can add another module to it.
        page = self.get("/developer/new").text
        self.assertIn('"name": "Decisions under Ambiguity"', page)
        form = self.new_element_form(display_name="Ambiguity Toy Two", setting="Decisions under Ambiguity",
                                     module="Ellsberg Choices")
        self.assertEqual(self.client.post("/developer/new", data=form, follow_redirects=False).status_code, 303)
        form = self.new_element_form(display_name="Ambiguity Toy Three", setting="Decisions under Ambiguity",
                                     module="__new__", new_module="Ambiguity Premia")
        self.assertEqual(self.client.post("/developer/new", data=form, follow_redirects=False).status_code, 303)
        # A new module can sit under an existing paper setting.
        setting = site.tree("STEER-ME")[1]["name"]
        form = self.new_element_form(display_name="Toy Four", setting=setting, module="__new__", new_module="Toy Module")
        self.assertEqual(self.client.post("/developer/new", data=form, follow_redirects=False).status_code, 303)

    def test_new_setting_and_module_are_checked(self):
        existing = site.tree("STEER-ME")[1]
        cases = [
            (dict(setting="__new__", module="__new__", new_setting="", new_module="M"), "Name the new setting"),
            (dict(setting="__new__", module="__new__", new_setting=existing["name"].upper(), new_module="M"), "already exists"),
            (dict(setting="__new__", module="", new_setting="Fresh Setting"), "needs a new module"),
            (dict(setting=existing["name"], module="__new__", new_module=existing["modules"][0]["name"]), "already exists"),
            (dict(setting=existing["name"], module="__new__", new_module=""), "Name the new module"),
        ]
        for overrides, message in cases:
            response = self.client.post("/developer/new", data=self.new_element_form(**overrides), follow_redirects=False)
            self.assertEqual(response.status_code, 400, overrides)
            self.assertIn(message, response.text, overrides)

    def test_discard_an_empty_draft(self):
        self.client.post("/developer/new", data=self.new_element_form(), follow_redirects=False)
        response = self.client.post("/developer/e/toy_surplus_element/discard", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertFalse((Path(self.tempdir.name) / "devmode" / "elements" / "toy_surplus_element.json").exists())
        self.assertEqual(self.get("/developer/e/toy_surplus_element").status_code, 404)

    # Setup and providers -------------------------------------------------------------------------------------
    def test_setup_reports_keys_without_showing_them(self):
        secret = "sk-test-DO-NOT-SHOW-0123456789"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret, "ANTHROPIC_API_KEY": secret + "a"}):
            page = self.get("/developer/setup").text
            candidates = self.get(f"/developer/e/{CODE_ELEMENT}").text
        self.assertNotIn(secret, page)
        self.assertNotIn(secret, candidates)
        for label in ("OpenAI", "Azure OpenAI", "Anthropic", "Google Gemini", "OpenAI-compatible endpoint"):
            self.assertIn(label, page)
        self.assertIn("<code>OPENAI_API_KEY</code> <span class=\"dev-ok\">set</span>", page)
        self.assertIn("<code>GEMINI_API_KEY</code> <span class=\"muted\">not set</span>", page)

    def test_setup_saves_default_models(self):
        response = self.client.post("/developer/setup", data={"gen_provider": "anthropic", "gen_model": "claude-sonnet-5-5",
                                                              "solve_provider": "openai", "solve_model": "gpt-5.4-mini"})
        self.assertEqual(response.status_code, 200)
        settings = json.loads((Path(self.tempdir.name) / "devmode" / "settings.json").read_text())
        self.assertEqual(settings["generation"], {"provider": "anthropic", "model": "claude-sonnet-5-5"})
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}):
            status = common_module._developer_status()
            choices = devmode.model_choices(status, self.store, "generation")
        self.assertEqual(choices["default"], "anthropic:claude-sonnet-5-5")

    def test_resolve_model(self):
        self.assertEqual(devmode.resolve_model("gemini:gemini-2.5-pro", {}), ("GEMINI", "gemini-2.5-pro"))
        self.assertEqual(devmode.resolve_model("openai_compatible:llama", {}), ("OPENAI_COMPATIBLE", "llama"))
        self.assertEqual(devmode.resolve_model("gpt-5.4-mini", {"provider": "Azure OpenAI"}), ("AZURE", "gpt-5.4-mini"))
        self.assertEqual(devmode.resolve_model("", {"provider": "OpenAI", "default_model": "gpt-5.4"}), ("OPENAI", "gpt-5.4"))


class ProviderClientTests(unittest.TestCase):
    """The new providers are constructed without calling any API."""

    def test_gemini_and_openai_compatible_use_the_openai_client_with_a_base_url(self):
        from autosteer.llm_client import LanguageModelClient

        with patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
            client = LanguageModelClient(provider_name="GEMINI")
        self.assertIn("generativelanguage.googleapis.com", str(client.provider.handler.base_url))
        with patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8000/v1"}, clear=False):
            client = LanguageModelClient(provider_name="OPENAI_COMPATIBLE")
        self.assertIn("localhost:8000", str(client.provider.handler.base_url))

    def test_anthropic_request_moves_system_prompt_and_joins_text(self):
        from autosteer.llm_client import LanguageModelClient

        client = LanguageModelClient(provider_name="ANTHROPIC", api_key="x")
        handler = MagicMock()
        handler.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"a": 1}')])
        client.provider.handler = handler
        text = client.make_request(user_prompts="Hi", system_prompts="Be brief.", request_type="completion",
                                   return_type="string", response_format={"type": "json_object"}, model="claude-haiku-4-5")
        self.assertEqual(text, '{"a": 1}')
        kwargs = handler.messages.create.call_args.kwargs
        self.assertIn("Be brief.", kwargs["system"])
        self.assertNotIn("response_format", kwargs)
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in kwargs["messages"]))


if __name__ == "__main__":
    unittest.main()
