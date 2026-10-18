"""Render exact captured terminal text as a polished macOS-style window PNG (2x)."""
import sys, os, textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter
S = 2
MONO = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15 * S)
_inter = os.path.expanduser("~/.fonts/Inter-Medium.ttf")
TITLE = ImageFont.truetype(_inter if os.path.exists(_inter) else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13 * S)
def color(l):
    if l.startswith("$"): return "#7DD3FC"
    if "True" in l or "BLUEBIRD" in l: return "#FCA5A5"
    if "False" in l: return "#5EEAD4"
    if l.startswith("==="): return "#CBD5E1"
    return "#E2E8F0"
def render(text, out, title, cols=96):
    lines = []
    for l in text.rstrip("\n").splitlines(): lines += textwrap.wrap(l, cols) or [""]
    cw = MONO.getbbox("M")[2]; lh = 24 * S; bar = 38 * S; pad = 26 * S; margin = 60 * S
    W = cols * cw + 2 * pad; H = bar + 2 * pad + len(lines) * lh - 6 * S
    im = Image.new("RGBA", (W + 2 * margin, H + 2 * margin), "white")
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([margin, margin + 14 * S, margin + W, margin + H + 14 * S], 12 * S, fill=(15, 23, 42, 70))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(22 * S)))
    d = ImageDraw.Draw(im); x0, y0 = margin, margin
    d.rounded_rectangle([x0, y0, x0 + W, y0 + H], 12 * S, fill="#0F172A")
    d.rounded_rectangle([x0, y0, x0 + W, y0 + bar], 12 * S, fill="#1E293B"); d.rectangle([x0, y0 + bar - 12 * S, x0 + W, y0 + bar], fill="#1E293B")
    d.line([x0, y0 + bar, x0 + W, y0 + bar], fill="#334155", width=S)
    for i, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        cx = x0 + 20 * S + i * 20 * S; cy = y0 + bar // 2; d.ellipse([cx - 6 * S, cy - 6 * S, cx + 6 * S, cy + 6 * S], fill=c)
    tw = d.textlength(title, font=TITLE); d.text((x0 + (W - tw) / 2, y0 + bar / 2), title, font=TITLE, fill="#94A3B8", anchor="lm")
    for i, l in enumerate(lines): d.text((x0 + pad, y0 + bar + pad + i * lh), l, font=MONO, fill=color(l))
    im.convert("RGB").save(out)
if __name__ == "__main__": render(open(sys.argv[1]).read(), sys.argv[2], sys.argv[3])
