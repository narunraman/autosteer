"""Write the README logos from the site's logo macros (templates/_logos.html), so they match the site header:
the STEER badge and the autoSTEER wheel (link icons), and the STEER lockup (badge + wordmark, as in the site header)
and the autoSTEER lockup (wheel + wordmark, as in the developer-mode header) for GitHub's light and dark themes.

    uv run python scripts/make_readme_logos.py <out-dir>
"""
import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
templates = Path(__file__).resolve().parents[1] / "src/autosteer/web/templates"
logos = Environment(loader=FileSystemLoader(templates)).get_template("_logos.html").module
THEMES = {"light": {"main": "#1f2328", "auto": "#5d6670"}, "dark": {"main": "#e8eaed", "auto": "#a2a9b0"}}
WORD_H, GAP = 16.55, 9                                # the header's sizes (site.css: .logo-word, .brand gap)
MARK_H = {"steer": 22.6, "autosteer": 28}              # .logo-badge, .logo-wheel


def parts(svg):
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    inner = svg[svg.index(">") + 1: svg.rindex("</svg>")]
    return vb, inner


def nested(svg, x, y, w, h):
    vb, inner = parts(svg)
    return f'<svg x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" viewBox="{" ".join(f"{v:g}" for v in vb)}">{inner}</svg>'


def paint(svg, colors):
    return svg.replace('class="logo-main"', f'fill="{colors["main"]}"').replace('class="logo-auto"', f'fill="{colors["auto"]}"')


badge = str(logos.badge("r"))
(out / "steer_badge.svg").write_text(
    badge.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(' class="logo-badge" aria-hidden="true"', ' role="img" aria-label="STEER"') + "\n")
wheel = str(logos.wheel("r"))
(out / "autosteer_wheel.svg").write_text(
    wheel.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(' class="logo-wheel" aria-hidden="true"', ' role="img" aria-label="autoSTEER"') + "\n")
for name, mark, macro in (("steer", badge, logos.steer), ("autosteer", wheel, logos.autosteer)):
    mvb, _ = parts(mark)
    mh = MARK_H[name]; mw = mh * mvb[2] / mvb[3]
    for theme, colors in THEMES.items():
        word = paint(str(macro()), colors)
        wvb, _ = parts(word)
        ww = WORD_H * wvb[2] / wvb[3]
        W, H = mw + GAP + ww, mh
        label = "STEER" if name == "steer" else "autoSTEER"
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" width="{W * 4:.0f}" height="{H * 4:.0f}" role="img" aria-label="{label}">'
               + nested(mark, 0, 0, mw, mh) + nested(word, mw + GAP, (H - WORD_H) / 2, ww, WORD_H) + "</svg>\n")
        (out / f"{name}_lockup_{theme}.svg").write_text(svg)
print("wrote", sorted(p.name for p in out.glob("*.svg")))
