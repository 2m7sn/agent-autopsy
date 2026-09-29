"""One self-contained HTML report: headline numbers, evolution chart, causes of death, trace, diff."""
import difflib
import json
from html import escape

CSS = """
:root{--bg:#0b0d10;--card:#13171c;--line:#232a33;--fg:#e6edf3;--mute:#8b98a5;--bad:#ff6b6b;--good:#3ddc97;--acc:#7aa2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:40px 16px 80px}h1{font-size:34px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--mute);margin:44px 0 12px}
.sub{color:var(--mute)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;overflow-x:auto}
.big{font:600 40px/1 ui-monospace,monospace;letter-spacing:-.03em}.big .from{color:var(--bad)}.big .to{color:var(--good)}
.lbl{color:var(--mute);font-size:13px;margin-top:8px}table{width:100%;border-collapse:collapse}
td,th{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mute);font-weight:500;font-size:13px}
code,pre{font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}pre{margin:0;white-space:pre-wrap;word-break:break-word}
.tag{display:inline-block;padding:2px 8px;border-radius:99px;background:#2a1618;color:var(--bad);font:12px ui-monospace,monospace}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:720px){.two{grid-template-columns:1fr}}
.step{padding:6px 0;border-bottom:1px dashed var(--line)}.fail{color:var(--bad)}.pass{color:var(--good)}
blockquote{margin:12px 0;padding:8px 12px;border-left:3px solid var(--acc);white-space:pre-wrap}.add{background:#0f2a1f;color:var(--good)}.del{background:#2a1618;color:var(--bad)}svg text{fill:var(--mute);font:11px ui-monospace,monospace}
"""


def _chart(history):
    w, h, pad = 960, 260, 36
    n = max(len(history) - 1, 1)
    x = lambda g: pad + (w - 2 * pad) * g / n
    y = lambda s: h - pad - (h - 2 * pad) * s
    grid = "".join(f'<line x1="{pad}" x2="{w-pad}" y1="{y(v)}" y2="{y(v)}" stroke="#232a33"/>'
                   f'<text x="4" y="{y(v)+4}">{v:.0%}</text>' for v in (0, .25, .5, .75, 1))
    dots = "".join(f'<circle cx="{x(hs["gen"])}" cy="{y(c["score"])}" r="4" fill="#7aa2ff" opacity=".45">'
                   f'<title>{escape(c["strategy"])}: {c["score"]:.0%}</title></circle>'
                   for hs in history for c in hs["candidates"])
    pts = " ".join(f"{x(hs['gen'])},{y(hs['best'])}" for hs in history)
    best = "".join(f'<circle cx="{x(hs["gen"])}" cy="{y(hs["best"])}" r="6" fill="#3ddc97"/>'
                   f'<text x="{x(hs["gen"])-10}" y="{h-10}">gen {hs["gen"]}</text>' for hs in history)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Success rate per generation">{grid}'
            f'<polyline points="{pts}" fill="none" stroke="#3ddc97" stroke-width="2.5"/>{dots}{best}</svg>')


def _trace(t):
    rows = f'<blockquote class="sub">{escape(t["msg"])}</blockquote>' + "".join(f'<div class="step"><code>{escape(s["tool"])}({escape(json.dumps(s["input"]))})</code>'
                   f'<br><code class="sub">→ {escape(json.dumps(s["output"], default=str)[:160])}</code></div>'
                   for s in t["steps"])
    verdict = ("".join(f'<div class="fail">✗ {escape(f)}</div>' for f in t["fails"])
               or '<div class="pass">✓ passed every check</div>')
    return rows + f'<div style="margin-top:10px">{verdict}</div>'


def _diff(a, b):
    out = []
    for line in difflib.ndiff(a.splitlines(), b.splitlines()):
        if line.startswith("?"):
            continue
        cls = {"+": "add", "-": "del"}.get(line[0], "")
        out.append(f'<div class="{cls}">{escape(line)}</div>')
    return "<pre>" + "".join(out) + "</pre>"


def render(res, mode):
    b, a = res["baseline_spec"], res["best_spec"]
    tools = "".join(f"<h3><code>{n}</code></h3>{_diff(b['tools'][n], a['tools'][n])}"
                    for n in b["tools"] if b["tools"][n] != a["tools"][n])
    causes = "".join(f'<tr><td><span class="tag">{escape(d["category"])}</span></td><td>{d["count"]}</td>'
                     f'<td>{escape(d["root_cause"])}</td><td>{escape(d["fix"])}</td></tr>' for d in res["diagnosis"])
    sample = res["sample"]
    trace = (f'<p class="sub">Holdout ticket <code>{sample["before"]["task"]}</code>, same model, same ticket{" (attack style never seen in training)" if sample["before"]["task"].startswith("x") else ""}.</p>'
             f'<div class="two"><div class="card"><b class="fail">Before</b>{_trace(sample["before"])}</div>'
             f'<div class="card"><b class="pass">After</b>{_trace(sample["after"])}</div></div>') if sample else ""
    num2 = lambda v, label: (f'<div class="card"><div class="big"><span class="from">{v["before"]:.0%}</span>'
                             f' → <span class="to">{v["after"]:.0%}</span></div><div class="lbl">{label}</div></div>')
    num = lambda k, label: num2(res[k], label)
    kinds = res.get("holdout_by_kind", {})
    extra = (num2(kinds["attacks"], "resisted attacks in styles the lab never trained on")
             + num2(kinds["tickets"], "normal holdout tickets still solved")) if "attacks" in kinds else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent Autopsy Report</title><style>{CSS}</style></head>
<body><main><h1>Agent Autopsy Report</h1>
<p class="sub">{escape(b.get("name", "agent"))} · mode: {mode} · {len(res["history"]) - 1} generations · winner: <code>{escape(res["best_strategy"])}</code></p>
<div class="grid">{num("holdout", "success on holdout tickets the lab never saw")}{num("train", "success on training tickets")}
{extra}<div class="card"><div class="big">{len(res["diagnosis"])}</div><div class="lbl">distinct causes of death found and repaired</div></div></div>
<h2>Evolution</h2><div class="card">{_chart(res["history"])}<p class="sub">Green = best so far. Blue = competing surgeon variants per generation.</p></div>
<h2>Causes of death</h2><div class="card"><table><tr><th>Cause</th><th>Hits</th><th>Root cause</th><th>Fix</th></tr>{causes}</table></div>
<h2>Same ticket, before and after</h2>{trace}
<h2>The surgery (diff)</h2><div class="card"><h3>System prompt</h3>{_diff(b["system"], a["system"])}{tools}</div>
</main></body></html>"""
