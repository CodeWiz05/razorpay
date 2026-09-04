import textwrap

W = 1240
parts = []

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def box(x, y, w, h, fill, stroke, rx=10, stroke_width=2, dash=None):
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dasharray}/>')

def text(x, y, s, size=15, weight="normal", anchor="middle", fill="#0f172a", family="monospace", style="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{fill}" font-family="{family}" font-style="{style}">{esc(s)}</text>')

def arrow(x1, y1, x2, y2, color="#475569", dash=None, width=2.2):
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{width}" marker-end="url(#arrowhead)"{dasharray}/>')

def code_box(x, y, w, h, title, bullets=None, sub=None, title_size=15, bsize=11):
    box(x, y, w, h, "#f1f5f9", "#334155")
    text(x + w/2, y + 25, title, size=title_size, weight="bold", fill="#0f172a")
    cy = y + 46
    if sub:
        text(x + w/2, cy, sub, size=11, fill="#64748b", style="italic")
        cy += 20
    if bullets:
        for b in bullets:
            text(x + 16, cy, f"- {b}", size=bsize, anchor="start", fill="#334155")
            cy += 16.5

def data_box(x, y, w, h, title, sub=None):
    box(x, y, w, h, "#dbeafe", "#2563eb")
    text(x + w/2, y + h/2 - (6 if sub else -4), title, size=14, weight="bold", fill="#1e3a8a")
    if sub:
        text(x + w/2, y + h/2 + 14, sub, size=10.5, fill="#1e40af")

def output_box(x, y, w, h, lines):
    box(x, y, w, h, "#dcfce7", "#059669")
    cy = y + h/2 - (len(lines)-1)*8 + 4
    for line in lines:
        text(x + w/2, cy, line, size=11.5, weight="bold", fill="#065f46")
        cy += 17

def demo_box(x, y, w, h, title, bullets=None):
    box(x, y, w, h, "#ede9fe", "#7c3aed")
    text(x + w/2, y + 23, title, size=13.5, weight="bold", fill="#4c1d95")
    cy = y + 44
    if bullets:
        for b in bullets:
            text(x + 16, cy, f"- {b}", size=11, anchor="start", fill="#5b21b6")
            cy += 16.5

def side_note(x, y, w, h, lines, fill, stroke, textfill, dash="4,3"):
    box(x, y, w, h, fill, stroke, dash=dash)
    cy = y + h/2 - (len(lines)-1)*7 + 4
    for line in lines:
        text(x + w/2, cy, line, size=10.5, fill=textfill)
        cy += 15

# ---------------------------------------------------------------- columns
LX, LW = 50, 500          # primary column: 50 -> 550
GX, GW = 562, 210         # side-note gap column: 562 -> 772
RX, RW = 786, 404         # secondary column: 786 -> 1190

# ---------------------------------------------------------------- header
parts_header_y = 34
text(W/2, 34, "Return-Risk Scorer -- System Architecture", size=22, weight="bold", family="sans-serif")
text(W/2, 55, "Razorpay AI Buildathon -- AI Risk Manager track", size=13, fill="#64748b", family="sans-serif")

box(LX-10, 76, LW+20, 28, "#eff6ff", "#93c5fd", rx=6, stroke_width=1.5)
text(LX + LW/2, 95, "PRIMARY ARTIFACT -- synthetic India D2C data", size=12.5, weight="bold", family="sans-serif", fill="#1e3a8a")

box(RX-10, 76, RW+20, 28, "#f5f3ff", "#c4b5fd", rx=6, stroke_width=1.5)
text(RX + RW/2, 95, "SECONDARY ARTIFACT -- real fraud data (ULB)", size=12.5, weight="bold", family="sans-serif", fill="#4c1d95")

# ---------------------------------------------------------------- PRIMARY column
y = 122
code_box(LX, y, LW, 50, "generate_data.py", sub="synthetic 60k-order generator, cited India benchmarks")
y2 = y + 50
arrow(LX+LW/2, y2, LX+LW/2, y2+20)
y = y2 + 20

data_box(LX, y, LW, 46, "data_orders.csv", sub="60,000 orders, 12-month window")
data_y_end = y + 46

side_note(GX, y-2, GW, 50, ["verify_leakage_", "invariant.py", "(independent check)"], "#fef9c3", "#ca8a04", "#713f12")
arrow(LX+LW, y+23, GX, y+23, color="#ca8a04", dash="4,3")

arrow(LX+LW/2, data_y_end, LX+LW/2, data_y_end+20)
y = data_y_end + 20

code_box(LX, y, LW, 44, "features.py", sub="leakage-safe feature matrix (expanding-window)")
y2 = y + 44
arrow(LX+LW/2, y2, LX+LW/2, y2+20)
y = y2 + 20

train_bullets = [
    "temporal split: train 8mo / val 2mo / test 2mo",
    "rule baseline + LogReg (interpretable reference)",
    "HGB, damped segment-weighted -- PRODUCTION model",
    "XGBoost -- comparison only",
    "calibrate (isotonic) BEFORE threshold search",
    "  [ordering fix -- see README Sec. 2]",
    "macro-cost threshold selection, not blended",
    "  (blended cost favors majority segment)",
    "single frozen TEST evaluation + MCC",
    "reason codes: permutation importance, direction-aware",
    "recall-by-segment diagnostics (COD vs Prepaid)",
]
train_h = 46 + len(train_bullets)*16.5 + 10
code_box(LX, y, LW, train_h, "train.py", title_size=16, bullets=train_bullets)
y2 = y + train_h
arrow(LX+LW/2, y2, LX+LW/2, y2+20)
y = y2 + 20

out1_lines = ["model.joblib  -  calibrator.joblib", "summary.json  -  reason_code_reference.json"]
output_box(LX, y, LW, 46, out1_lines)
out_y_end = y + 46
y = out_y_end + 20

side_note(GX, out_y_end-4, GW, 54, ["evaluate.py", "-> evaluation_plots.png", "(PR / cost / calibration)"], "#f1f5f9", "#334155", "#0f172a")
arrow(LX+LW, out_y_end+13, GX, out_y_end+13, color="#334155", dash="4,3")

arrow(LX+LW/2, out_y_end, LX+LW/2, out_y_end+20)

serve_bullets = [
    "DEFENSE-ONLY: two response shapes, one endpoint",
    "  (reviewer_view flag)",
    "reviewer_view=False (default): checkout-facing --",
    "  coarse decision + reason codes. No score, no",
    "  threshold. Prevents decision-boundary probing.",
    "reviewer_view=True: full transparency (score,",
    "  calibrated_score, threshold) -- internal only.",
]
serve_h = 46 + len(serve_bullets)*16.5 + 10
box(LX-6, y-6, LW+12, serve_h+12, "none", "#dc2626", rx=14, stroke_width=2, dash="6,3")
code_box(LX, y, LW, serve_h, "serve.py  (FastAPI)", title_size=15, bullets=serve_bullets)
text(LX+LW-8, y+serve_h+3, "defense-only boundary", size=9.5, anchor="end", fill="#dc2626", style="italic", family="sans-serif")
y2 = y + serve_h
arrow(LX+LW/2, y2+8, LX+LW/2, y2+26)
y = y2 + 26

demo_h = 70
demo_box(LX, y, LW, demo_h, "app.py  (Streamlit reviewer demo)", bullets=[
    "shows BOTH response views side by side, live",
    "presets incl. bracketed-order example (pitch video)",
])
primary_end_y = y + demo_h

# ---------------------------------------------------------------- SECONDARY column
y = 122
data_box(RX, y, RW, 50, "creditcard.csv", sub="ULB Kaggle, 284,807 real anonymized txns")
y2 = y + 50
arrow(RX+RW/2, y2, RX+RW/2, y2+20)
y = y2 + 20

tf_bullets = [
    "proportional 60/20/20 split by Time",
    "  (no calendar structure available)",
    "HGB vs XGBoost on VAL -- XGBoost wins,",
    "  selected as primary here",
    "calibrate BEFORE threshold search",
    "  [same ordering fix as train.py]",
    "single frozen TEST evaluation",
    "NO reason codes (V1-V28 PCA-anonymized)",
    "NO customer entity -- nothing to leak the",
    "  primary's expanding-window bug against",
]
tf_h = 46 + len(tf_bullets)*16.5 + 10
code_box(RX, y, RW, tf_h, "train_fraud.py", title_size=15, bullets=tf_bullets, bsize=10.5)
y2 = y + tf_h
arrow(RX+RW/2, y2, RX+RW/2, y2+20)
y = y2 + 20

output_box(RX, y, RW, 46, ["model_fraud.joblib  -  summary_fraud.json", "val / test probability arrays"])
out2_end = y + 46
y = out2_end + 20

arrow(RX+RW/2, out2_end, RX+RW/2, out2_end+20)
code_box(RX, y, RW, 44, "evaluate_fraud.py", sub="-> evaluation_plots_fraud.png")
y2 = y + 44
y = y2 + 20

note_h = 54
side_note(RX, y, RW, note_h, ["No serving layer for this artifact --", "comparison / methodology-breadth only"],
          "#fef2f2", "#fca5a5", "#991b1b", dash="4,3")
secondary_end_y = y + note_h

# ---------------------------------------------------------------- legend
content_end = max(primary_end_y, secondary_end_y)
leg_y = content_end + 46
text(W/2, leg_y-16, "Legend", size=13, weight="bold", family="sans-serif")
items = [
    ("#f1f5f9", "#334155", "code / script"),
    ("#dbeafe", "#2563eb", "data file"),
    ("#dcfce7", "#059669", "saved model / metric artifact"),
    ("#ede9fe", "#7c3aed", "serving / demo layer"),
    ("#fef9c3", "#ca8a04", "independent verification"),
]
total_w = sum(20 + 8 + len(l)*6.5 + 24 for _,_,l in items)
lx = (W - total_w) / 2
for fill, stroke, label in items:
    box(lx, leg_y, 18, 18, fill, stroke, rx=4, stroke_width=1.5)
    text(lx+26, leg_y+14, label, size=11, anchor="start", fill="#334155", family="sans-serif")
    lx += 20 + 8 + len(label)*6.5 + 24

text(W/2, leg_y+40, "Full methodology, citations, and negative results: README Sections 2-6", size=11, fill="#64748b", family="sans-serif", style="italic")

H = int(leg_y + 66)

svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="monospace">
<defs>
<marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
<polygon points="0 0, 8 4, 0 8" fill="#475569"/>
</marker>
</defs>
<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>
{"".join(parts)}
</svg>'''

with open("C:/Numair/Coding/Razorpay/architecture.svg", "w") as f:
    f.write(svg)

print("H =", H)
print("written, length:", len(svg))