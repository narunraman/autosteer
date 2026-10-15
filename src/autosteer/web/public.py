"""What the hosted site adds around the pages: launch switches, robots.txt and sitemap, a 404 page, and limits on
/api/sample.

Switches (environment variables, read on every request):

``AUTOSTEER_EVALS``
    ``on`` or ``off``. Unset: off on the public site (``AUTOSTEER_WEB_MODE=public``), on in local mode. Off hides
    the Evals section entirely: the header link, the pages (404), the scored results under /static, and the
    links to them from other pages.
``AUTOSTEER_SITE_URL``
    The public address, for canonical links, social previews and the sitemap.
``AUTOSTEER_API_RATE``
    Requests to /api/sample allowed per client address per minute. Unset: 60 on the public site, no limit in
    local mode; 0 turns the limit off. The reverse proxy should limit as well.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Iterable

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from xml.sax.saxutils import escape as xml_escape

DEFAULT_SITE_URL = "https://steer-benchmark.cs.ubc.ca"
TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}

# Served by the app but only part of the site while Evals is shown.
EVALS_STATIC = ("/static/evals.js", "/static/evals_results.json")
# Developer views that the public site does not serve: the raw records of an element's seeds and templates,
# and the generated API docs (which list the authoring endpoints).
_DEV_PREFIXES = ("/api/datasets/", "/docs", "/redoc", "/openapi.json")

API_MAX_QUERY = 2048  # bytes of query string
API_MAX_BODY = 16 * 1024  # bytes of request body (POST /api/sample takes a small JSON object)
API_CACHE_SIZE = 512  # seeded /api/sample responses kept in memory
API_CACHE_MAX_BYTES = 512 * 1024  # larger responses are not kept


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value in TRUE:
        return True
    if value in FALSE:
        return False
    return default


def evals_enabled(public: bool) -> bool:
    return _flag("AUTOSTEER_EVALS", not public)


def site_url() -> str:
    return (os.getenv("AUTOSTEER_SITE_URL") or DEFAULT_SITE_URL).rstrip("/")


def api_rate(public: bool) -> int:
    value = os.getenv("AUTOSTEER_API_RATE", "").strip()
    if value.isdigit():
        return int(value)
    return 60 if public else 0


def _is_dev_only(path: str) -> bool:
    if path.startswith(_DEV_PREFIXES):
        return True
    parts = [p for p in path.split("/") if p]
    # /elements/<element>/<kind>: the developer view of an element's records (the element page is /elements/<element>)
    return len(parts) == 3 and parts[0] == "elements"


def _wants_html(request: Request) -> bool:
    path = request.url.path
    if path.startswith(("/api/", "/static/")):
        return False
    return "text/html" in request.headers.get("accept", "")


class _RateLimiter:
    """At most ``limit`` requests per client in any 60 seconds (in one process; run the site as one process)."""

    def __init__(self) -> None:
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()
        self.swept = time.monotonic()

    def retry_after(self, client: str, limit: int) -> int:
        """0 if the request may go ahead (and count it), else seconds until it may."""
        now = time.monotonic()
        with self.lock:
            if now - self.swept > 300:
                self.hits = {k: q for k, q in self.hits.items() if q and now - q[-1] < 60}
                self.swept = now
            q = self.hits.setdefault(client, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            return 0


class _ResponseCache:
    """The responses to seeded GET /api/sample requests, which are deterministic: same query, same body."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)


def install(
    app: FastAPI,
    templates: Jinja2Templates,
    public_mode: Callable[[], bool],
    sitemap_paths: Callable[[], Iterable[str]],
) -> None:
    limiter = _RateLimiter()
    cache = _ResponseCache(API_CACHE_SIZE)
    templates.env.globals.update(
        show_evals=lambda: evals_enabled(public_mode()),
        site_url=site_url,
    )

    def not_found(request: Request, detail: str = "Not Found") -> Response:
        if _wants_html(request):
            return templates.TemplateResponse(
                request, "404.html",
                {"page": "", "group_title": "Not found", "page_label": "Not found", "page_title": "Page not found",
                 "noindex": True},
                status_code=404,
            )
        return JSONResponse({"detail": detail}, status_code=404)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404 and _wants_html(request):
            return not_found(request)
        return await http_exception_handler(request, exc)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        public = public_mode()
        if public and _is_dev_only(path):
            return not_found(request)
        if path in EVALS_STATIC and not evals_enabled(public):
            return not_found(request)
        if path.rstrip("/") != "/api/sample":
            return await call_next(request)

        # /api/sample: bounded requests, a per-client rate, and seeded answers cached.
        if len(request.url.query.encode()) > API_MAX_QUERY:
            return JSONResponse({"detail": "Query string too long"}, status_code=414)
        length = request.headers.get("content-length", "")
        if (length.isdigit() and int(length) > API_MAX_BODY) or (request.method == "POST" and not length.isdigit()):
            return JSONResponse({"detail": f"Request body must be under {API_MAX_BODY} bytes"}, status_code=413)
        limit = api_rate(public)
        if limit:
            client = request.client.host if request.client else "unknown"
            wait = limiter.retry_after(client, limit)
            if wait:
                return JSONResponse({"detail": f"Too many requests: at most {limit} a minute. Use n (up to 50) to get "
                                               "several questions in one request."},
                                    status_code=429, headers={"Retry-After": str(wait)})

        seeded = request.method == "GET" and "seed" in request.query_params
        key = tuple(sorted(request.query_params.multi_items())) if seeded else None
        if key is not None:
            hit = cache.get(key)
            if hit is not None:
                return Response(hit, media_type="application/json",
                                headers={"Cache-Control": "public, max-age=86400", "X-Cache": "hit"})
        response = await call_next(request)
        if key is None or response.status_code != 200:
            response.headers["Cache-Control"] = "no-store"
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        if len(body) <= API_CACHE_MAX_BYTES:
            cache.put(key, body)
        return Response(body, status_code=200, media_type="application/json",
                        headers={"Cache-Control": "public, max-age=86400", "X-Cache": "miss"})

    @app.get("/robots.txt", include_in_schema=False)
    def robots() -> PlainTextResponse:
        return PlainTextResponse(
            "User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: " + site_url() + "/sitemap.xml\n"
        )

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap() -> Response:
        base = site_url()
        urls = "".join(f"  <url><loc>{xml_escape(base + path)}</loc></url>\n" for path in sitemap_paths())
        body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n")
        return Response(body, media_type="application/xml")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse(url="/static/favicon.svg", status_code=301)
