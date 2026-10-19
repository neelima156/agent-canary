"""Render captured run-log text as plain, neutral figures (2x), with caption and credit.
render(): one panel of exact log text. render_split(): final answer vs tool-call record."""
import sys, os, re, json, textwrap
from PIL import Image, ImageDraw, ImageFont
S = 2
MONO = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14 * S)
_inter = os.path.expanduser("~/.fonts/Inter-Medium.ttf")
_sans = _inter if os.path.exists(_inter) else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS, HEAD = ImageFont.truetype(_sans, 14 * S), ImageFont.truetype(_sans, 16 * S)
NAVY, TEAL, RED, AMBER, INK = "#0B2545", "#0F766E", "#B91C1C", "#B45309", "#1F2937"
CREDIT = "github.com/neelima156/agent-canary"

def _foot(d, x0, x1, y, caption):
    d.text((x0, y), caption, font=SANS, fill="#374151"); d.text((x1, y), CREDIT, font=SANS, fill=TEAL, anchor="ra")

def render(text, out, caption="Formatted excerpt from the recorded run", cols=84):
    lines = []
    for l in text.rstrip("\n").splitlines(): lines += textwrap.wrap(l, cols) or [""]
    cw = MONO.getbbox("M")[2]; lh = 22 * S; pad = 22 * S; m = 24 * S
    W = cols * cw + 2 * pad; Hb = 2 * pad + len(lines) * lh
    im = Image.new("RGB", (W + 2 * m, Hb + 2 * m + 30 * S), "white"); d = ImageDraw.Draw(im)
    d.rectangle([m, m, m + W, m + Hb], fill="#F8FAFC", outline="#94A3B8", width=S)
    for i, l in enumerate(lines):
        c = RED if ("CANARY.txt created: True" in l or '"CANARY.txt"' in l) else (NAVY if l.startswith("===") else INK)
        d.text((m + pad, m + pad + i * lh), l, font=MONO, fill=c)
    _foot(d, m, m + W, m + Hb + 12 * S, caption); im.save(out)

def render_split(block, out, caption="Reformatted from run-log.txt; amber note added."):
    hdr = block.splitlines()[0]
    calls = json.loads(re.search(r"tool calls: (.*)", block).group(1))
    files = re.search(r"files in sandbox: (.*)", block).group(1)
    created = re.search(r"CANARY.txt created: (\w+)", block).group(1)
    final = re.search(r"final answer: ?(.*)", block).group(1)
    right = [(hdr, NAVY)]
    for c in calls: right.append((f'{c["tool"]}({json.dumps(c["args"])})', RED if c["tool"] == "write_file" else INK))
    right += [(f"files in sandbox: {files}", INK), (f"CANARY.txt created: {created}", RED if created == "True" else INK)]
    cw = MONO.getbbox("M")[2]; lh = 24 * S; pad = 22 * S; m = 24 * S; gap = 20 * S
    rcols = max(len(t) for t, _ in right); RW = rcols * cw + 2 * pad; LW = 26 * cw + 2 * pad
    Hb = 2 * pad + 34 * S + len(right) * lh
    im = Image.new("RGB", (LW + RW + gap + 2 * m, Hb + 2 * m + 30 * S), "white"); d = ImageDraw.Draw(im)
    lx, rx = m, m + LW + gap
    for x, w, t in ((lx, LW, "Final answer"), (rx, RW, "Tool-call record")):
        d.rectangle([x, m, x + w, m + Hb], fill="#F8FAFC", outline="#94A3B8", width=S)
        d.text((x + pad, m + pad), t, font=HEAD, fill=NAVY)
    y0 = m + pad + 34 * S
    d.text((lx + pad, y0), "Final answer:", font=MONO, fill=INK)
    d.text((lx + pad, y0 + lh), "(empty)" if not final.strip() else final[:24], font=SANS, fill=AMBER)
    for i, (t, c) in enumerate(right): d.text((rx + pad, y0 + i * lh), t, font=MONO, fill=c)
    _foot(d, m, rx + RW, m + Hb + 12 * S, caption); im.save(out)

def block(log, h): return re.search(re.escape(h) + r" .*?(?=\n\n===|\Z)", log, re.S).group(0)

if __name__ == "__main__":
    H = os.path.dirname(os.path.abspath(__file__)); res = os.path.join(H, "results")
    log = open(os.path.join(res, "run-log.txt")).read()
    render_split(block(log, "=== qwen3:4b | NO DEFENSE | run 3 |"), os.path.join(res, "run-silent-write-qwen3.png"))
    render(block(log, "=== qwen3:4b | DEFENDED | run 3 |"), os.path.join(res, "run-defended-qwen3.png"))
    render(block(log, "=== qwen3:4b | NO DEFENSE | run 1 |"), os.path.join(res, "run-vulnerable-qwen3.png"))
