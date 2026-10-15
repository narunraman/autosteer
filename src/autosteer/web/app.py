"""The web app: the public site, the JSON API and developer mode.

Routes live in autosteer.web.routes (one router per area), in autosteer.web.review, .devmode and .site; what they
share is in autosteer.web.common. Run with `autosteer dev`, or `uvicorn autosteer.web.app:app`.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from autosteer.sampler import load_manifest
from autosteer.web import common, devmode, review, site
from autosteer.web.routes import api, candidates, seeds, template_gen


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the largest elements' templates in the background, so the first visitor to one does not wait for it
    (shared by the API, the element pages and developer mode). AUTOSTEER_WARM=0 turns it off."""
    if os.environ.get("AUTOSTEER_WARM", "1") != "0":
        threading.Thread(target=lambda: common._api_sampler(load_manifest()).warm(), name="warm-templates",
                         daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="STEER Generation", lifespan=lifespan)
    common.templates.env.globals["no_provider_text"] = common.NO_PROVIDER
    app.mount("/static", StaticFiles(directory=str(common.WEB_DIR / "static")), name="static")
    # the order matters: routes are matched in the order they are added
    for area in (api, candidates, template_gen, seeds):
        app.include_router(area.router)
    review.install(app, common.templates, public_mode=common._public_web_mode)
    devmode.install(app, common.templates, public_mode=common._public_web_mode,
                    status=lambda: common._developer_status(), step_context=common.step_context)
    site.install(app, common.templates, public_mode=common._public_web_mode)
    return app


app = create_app()
