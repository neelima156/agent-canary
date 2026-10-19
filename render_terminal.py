"""Render exact captured run-log text in a plain neutral frame (2x), with caption and credit."""
import sys, os, textwrap
from PIL import Image, ImageDraw, ImageFont
S = 2
MONO = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15 * S)
_inter = os.path.expanduser("~/.fonts/Inter-Medium.ttf")
SANS = ImageFont.truetype(_inter if os.path.exists(_inter) else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14 * S)
NAVY, TEAL, RED, AMBER, INK = "#0B2545", "#0F766E", "#B91C1C", "#B45309", "#1F2937"
def color(l):
    if "CANARY.txt created: True" in l or '"CANARY.txt"' in l: return RED
    if l.startswith("==="): return NAVY
    return INK
def render(text, out, caption="Formatted excerpt from the recorded run", cols=92, notes=None):
    lines = []
    for l in text.rstrip("\n").splitlines(): lines += textwrap.wrap(l, cols) or [""]
    cw = MONO.getbbox("M")[2]; lh = 24 * S; pad = 26 * S; m = 40 * S
    W = cols * cw + 2 * pad; Hb = 2 * pad + len(lines) * lh
    im = Image.new("RGB", (W + 2 * m, Hb + m + 50 * S), "white"); d = ImageDraw.Draw(im)
    d.rectangle([m, m, m + W, m + Hb], fill="#F8FAFC", outline="#94A3B8", width=S)
    for i, l in enumerate(lines):
        y = m + pad + i * lh; d.text((m + pad, y), l, font=MONO, fill=color(l))
        if notes and l in notes:
            x = m + pad + d.textlength(l, font=MONO) + 12 * S; d.text((x, y + 2 * S), notes[l], font=SANS, fill=AMBER)
    yb = m + Hb + 14 * S
    d.text((m, yb), caption, font=SANS, fill="#374151")
    d.text((m + W, yb), "github.com/neelima156/agent-canary", font=SANS, fill=TEAL, anchor="ra")
    im.save(out)
if __name__ == "__main__": render(open(sys.argv[1]).read(), sys.argv[2])
