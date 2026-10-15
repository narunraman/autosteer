"""Render static/og.png (1200x630) and static/apple-touch-icon.png (180x180) for the site.

The badge and the wordmark come from templates/_logos.html, so the images match the header.

    uv run --with playwright python scripts/make_site_images.py src/autosteer/web/static
"""
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1])
templates = Path(__file__).resolve().parents[1] / "src/autosteer/web/templates"
logos = Environment(loader=FileSystemLoader(templates)).get_template("_logos.html").module
BADGE, STEER = str(logos.badge("og")), str(logos.steer())
SETTINGS = ("#c4262e", "#2e9d5b", "#2f6db5", "#d9a21b", "#7b4bb7")   # the five setting colors, top to bottom

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400&display=swap">'
OG = FONTS + """
<style>
 html,body{margin:0;width:1200px;height:630px;background:#fff;font-family:'Source Sans 3',sans-serif;color:#1f2328}
 .bar{position:absolute;left:0;top:0;bottom:0;width:22px;display:flex;flex-direction:column}
 .bar i{flex:1}
 .wrap{position:absolute;inset:0;padding:96px 100px;display:flex;flex-direction:column;justify-content:center}
 .brand{display:flex;align-items:center;gap:34px}
 .brand .logo-badge{height:136px;width:auto;display:block}
 .brand .logo-word{height:100px;width:auto;display:block}
 .brand .logo-main{fill:#1f2328}
 .sub{margin-top:52px;font-size:46px;line-height:1.25;color:#2f353b;max-width:980px}
</style>
<div class="bar">""" + "".join(f'<i style="background:{c}"></i>' for c in SETTINGS) + """</div>
<div class="wrap">
 <div class="brand">""" + BADGE + STEER + """</div>
 <div class="sub">Benchmarks for the economic reasoning of language models</div>
</div>"""
# iOS rounds the corners itself, so the icon is the plate edge to edge, letters centered
ICON = """
<style>html,body{margin:0;width:180px;height:180px;background:#c4262e;display:flex;align-items:center}
.logo-badge{width:180px;height:auto;display:block}</style>""" + BADGE.replace('viewBox="-1 -1 122 74"', 'viewBox="0 0 120 72"')

with sync_playwright() as p:
    b = p.chromium.launch()
    for html, name, (w, h) in ((OG, "og.png", (1200, 630)), (ICON, "apple-touch-icon.png", (180, 180))):
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.set_content(html, wait_until="networkidle")
        pg.evaluate("document.fonts.ready")
        pg.screenshot(path=str(out / name))
    b.close()
