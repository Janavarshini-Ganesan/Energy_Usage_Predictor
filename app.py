"""
app.py
------
WattWise: Gradio dashboard for household energy forecasting with Elastic Net Regression.

Run locally:
    python app.py

Deploy: Hugging Face Spaces (SDK: Gradio). If the saved model is missing or was pickled
with an incompatible scikit-learn version, it is retrained automatically on startup.
"""

import json
import os
import random
import subprocess
import sys

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model_elasticnet.pkl")
INFO_PATH = os.path.join(BASE, "model_info.json")
DATA_PATH = os.path.join(BASE, "data", "energy_usage.csv")


# ------------------------------------------------------------------ artifacts
def load_artifacts():
    try:
        with open(INFO_PATH) as f:
            return joblib.load(MODEL_PATH), json.load(f)
    except Exception:
        subprocess.run([sys.executable, "train.py"], cwd=BASE, check=True)
        with open(INFO_PATH) as f:
            return joblib.load(MODEL_PATH), json.load(f)


MODEL, INFO = load_artifacts()
SCALER = MODEL.named_steps["standardscaler"]
ENET = MODEL.named_steps["elasticnetcv"]
FEATURES = INFO["feature_names"]
COEF = np.array([INFO["elasticnet_coefs"][f] for f in FEATURES])
LASSO_COEF = INFO["lasso_coefs"]
METRICS = INFO["metrics"]

DF = pd.read_csv(DATA_PATH)
X_ALL, Y_ALL = DF[FEATURES], DF["Daily_Energy_kWh"]
X_TR, _, Y_TR, _ = train_test_split(X_ALL, Y_ALL, test_size=0.2, random_state=42)
X_TR_S = SCALER.transform(X_TR)
Q25, Q75, Q90 = Y_ALL.quantile([0.25, 0.75, 0.90])
Y_MEAN = float(Y_ALL.mean())

LABELS = {
    "Outdoor_Temp_C": "Outdoor temperature",
    "Humidity_Pct": "Humidity",
    "AC_Hours": "AC hours",
    "Heating_Hours": "Heating hours",
    "Occupants": "Occupants",
    "Home_Area_sqft": "Home area",
    "Num_Rooms": "Rooms",
    "Appliance_Count": "Appliances",
    "Neighbourhood_Noise_Index": "Neighbourhood noise",
    "Window_Color_Code": "Window colour",
    "Pet_Count": "Pets",
    "Roof_Age_Years": "Roof age",
    "Solar_Panel_Score": "Solar panel score",
}
GROUP = {
    **{f: "Climate" for f in ["Outdoor_Temp_C", "Humidity_Pct", "AC_Hours", "Heating_Hours"]},
    **{f: "Household" for f in ["Occupants", "Home_Area_sqft", "Num_Rooms", "Appliance_Count"]},
    **{f: "Unrelated" for f in ["Neighbourhood_Noise_Index", "Window_Color_Code", "Pet_Count",
                                "Roof_Age_Years", "Solar_Panel_Score"]},
}
GROUP_COLOR = {"Climate": "#22d3ee", "Household": "#f472b6", "Unrelated": "#6b7394"}
NEGLIGIBLE = 0.15  # |scaled coef| below this is treated as "no real effect"

CO2_KG_PER_KWH = 0.8  # rough grid-average assumption, shown as such in the UI


# ------------------------------------------------------------------ skyline art
def skyline_html() -> str:
    rnd = random.Random(11)
    parts, x = [], 0
    while x < 1000:
        w, h = rnd.randint(34, 70), rnd.randint(40, 118)
        shade = rnd.choice(["#141a3d", "#181f47", "#111634"])
        parts.append(f'<rect x="{x}" y="{160 - h}" width="{w}" height="{h}" fill="{shade}"/>')
        for wy in range(160 - h + 8, 150, 14):
            for wx in range(x + 6, x + w - 8, 12):
                if rnd.random() < 0.42:
                    col = rnd.choice(["#ffd166", "#5eead4", "#f472b6"])
                    delay = round(rnd.uniform(0, 5), 1)
                    parts.append(
                        f'<rect class="win" style="animation-delay:{delay}s" x="{wx}" y="{wy}" '
                        f'width="5" height="7" rx="1" fill="{col}"/>'
                    )
        x += w + rnd.randint(2, 8)
    svg = (
        '<svg class="skyline" viewBox="0 0 1000 160" preserveAspectRatio="xMidYMax slice" '
        'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )
    return f"""
    <div class="hero">
        <div class="moon"></div>
        {svg}
        <div class="hero-text">
            <div class="hero-kicker">ELASTIC NET REGRESSION · DAY 6</div>
            <div class="hero-title">Watt<span>Wise</span></div>
            <div class="hero-sub">Forecast a household's daily electricity use, and see which factors really drive it.</div>
        </div>
    </div>
    """


# ------------------------------------------------------------------ forecast
def _level(kwh: float):
    if kwh < Q25:
        return "Low usage", "#5eead4"
    if kwh < Q75:
        return "Typical usage", "#22d3ee"
    if kwh < Q90:
        return "High usage", "#fbbf24"
    return "Very high usage", "#f472b6"


def forecast(*vals):
    tariff = float(vals[-1])
    row = dict(zip(FEATURES, [float(v) for v in vals[:-1]]))
    X = pd.DataFrame([row])[FEATURES]
    daily = max(0.0, float(MODEL.predict(X)[0]))

    contrib = SCALER.transform(X)[0] * COEF
    order = np.argsort(-np.abs(contrib))[:6]
    top = float(np.abs(contrib[order]).max()) or 1.0

    driver_rows = ""
    for i in order:
        c = float(contrib[i])
        pct = max(4, abs(c) / top * 100)
        colour = "#f472b6" if c > 0 else "#22d3ee"
        driver_rows += (
            f'<div class="drv"><div class="drv-name">{LABELS[FEATURES[i]]}</div>'
            f'<div class="drv-track"><div class="drv-fill" style="width:{pct:.0f}%;background:{colour}"></div></div>'
            f'<div class="drv-val" style="color:{colour}">{c:+.1f}</div></div>'
        )

    frac = min(1.0, max(0.0, (daily - 6) / 36))
    arc_len = 251.3
    level, level_col = _level(daily)
    monthly = daily * 30

    return f"""
    <div class="res">
        <div class="res-gauge">
            <svg viewBox="0 0 200 118" width="100%">
                <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#f472b6"/></linearGradient></defs>
                <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#1d2350" stroke-width="14" stroke-linecap="round"/>
                <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#g)" stroke-width="14" stroke-linecap="round"
                      stroke-dasharray="{arc_len}" stroke-dashoffset="{arc_len * (1 - frac):.1f}"/>
                <text x="100" y="86" text-anchor="middle" class="g-num">{daily:.1f}</text>
                <text x="100" y="106" text-anchor="middle" class="g-unit">kWh per day</text>
            </svg>
            <div class="badge" style="color:{level_col};border-color:{level_col}55">{level}</div>
        </div>
        <div class="stat-grid">
            <div class="stat"><div class="stat-k">Per month</div><div class="stat-v">{monthly:,.0f} kWh</div></div>
            <div class="stat"><div class="stat-k">Est. bill</div><div class="stat-v">₹{monthly * tariff:,.0f}</div></div>
            <div class="stat"><div class="stat-k">CO₂ (approx.)</div><div class="stat-v">{monthly * CO2_KG_PER_KWH:,.0f} kg</div></div>
            <div class="stat"><div class="stat-k">vs. average home</div><div class="stat-v">{(daily / Y_MEAN - 1) * 100:+.0f}%</div></div>
        </div>
        <div class="drv-title">What's moving this forecast <span>(kWh/day vs. an average home)</span></div>
        {driver_rows}
    </div>
    """


# ------------------------------------------------------------------ model insights
def metrics_html() -> str:
    cv = INFO["cv_scores"]
    cv_rows = "".join(
        f'<div class="cvrow"><span>{k}</span><b>{v["mean"]:.3f} ± {v["std"]:.3f}</b></div>' for k, v in cv.items()
    )
    return f"""
    <div class="stat-grid four">
        <div class="stat"><div class="stat-k">Test R²</div><div class="stat-v">{METRICS['r2']:.3f}</div></div>
        <div class="stat"><div class="stat-k">Test MAE</div><div class="stat-v">{METRICS['mae']:.2f} kWh</div></div>
        <div class="stat"><div class="stat-k">alpha (CV)</div><div class="stat-v">{METRICS['alpha']:.3f}</div></div>
        <div class="stat"><div class="stat-k">l1_ratio</div><div class="stat-v">{METRICS['l1_ratio']}</div></div>
    </div>
    <div class="note">
        <b>Cross-validated R² on the training data</b>
        {cv_rows}
        <p>Accuracy is effectively a tie: the gap is far smaller than the fold-to-fold spread, so accuracy alone can't
        choose between the two. <code>l1_ratio</code> is fixed at 0.5 on purpose. The L2 half makes correlated features
        share weight instead of one being picked arbitrarily. Compare the bars (Elastic Net) with the diamonds (Lasso)
        below, especially for <i>Occupants</i> and <i>Home area</i>.</p>
    </div>
    """


def coef_plot(exp: float):
    alpha = 10 ** exp
    en = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000).fit(X_TR_S, Y_TR)
    coefs = dict(zip(FEATURES, en.coef_))
    order = sorted(FEATURES, key=lambda f: abs(INFO["elasticnet_coefs"][f]))

    fig = Figure(figsize=(7.4, 5.4), facecolor="#0f1330")
    ax = fig.subplots()
    ax.set_facecolor("#0f1330")
    ypos = np.arange(len(order))
    ax.barh(ypos, [coefs[f] for f in order], color=[GROUP_COLOR[GROUP[f]] for f in order], height=0.62, alpha=0.92)
    ax.scatter([LASSO_COEF[f] for f in order], ypos, marker="D", s=42, facecolor="#0f1330",
               edgecolor="#fbbf24", linewidth=1.6, zorder=3)
    ax.axvline(0, color="#3b4275", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS[f] for f in order], color="#c9d1f5", fontsize=9.5)
    ax.tick_params(axis="x", colors="#8b93c4", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("standardised coefficient (kWh/day per 1 SD)", color="#8b93c4", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color="#1d2350", lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=GROUP_COLOR[g], label=g) for g in GROUP_COLOR] + [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#0f1330", markeredgecolor="#fbbf24",
               markeredgewidth=1.6, label="Lasso (CV alpha)")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor="#c9d1f5")
    fig.tight_layout()

    small = sum(abs(v) < NEGLIGIBLE for v in coefs.values())
    exact0 = sum(v == 0 for v in coefs.values())
    summary = (
        f'<div class="note tight">alpha = <b>{alpha:.3f}</b> &nbsp;·&nbsp; '
        f'<b>{small}</b> of {len(FEATURES)} features have a negligible effect (|coef| &lt; {NEGLIGIBLE}) '
        f'&nbsp;·&nbsp; <b>{exact0}</b> are exactly zero</div>'
    )
    return fig, summary


# ------------------------------------------------------------------ styling
FONTS = "@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@300;400;500;600&display=swap');"

CSS = FONTS + """
:root, .dark, .gradio-container {
    --body-background-fill: #070a1a;
    --background-fill-primary: #0f1330;
    --background-fill-secondary: #0b0f26;
    --block-background-fill: #0f1330;
    --block-border-color: #222a5c;
    --border-color-primary: #222a5c;
    --body-text-color: #e6e9ff;
    --block-label-text-color: #8b93c4;
    --block-title-text-color: #a9b1e6;
    --input-background-fill: #0b0f26;
    --input-border-color: #2a3270;
    --color-accent: #22d3ee;
    --slider-color: #22d3ee;
    --button-primary-background-fill: linear-gradient(135deg, #22d3ee, #6366f1);
    --button-primary-text-color: #06091c;
}
body, .gradio-container { background: #070a1a !important; font-family: 'Outfit', sans-serif !important; }
.gradio-container {
    width: 100% !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
    box-sizing: border-box !important;
}

footer { opacity: .55; }

/* hero */
.hero {
    position: relative;
    overflow: hidden;
    width: 100%;
    box-sizing: border-box;
    border-radius: 22px;
    margin: 0 auto 14px auto;
    min-height: 250px;
    background: radial-gradient(ellipse at 78% 0%, #3b1d6e 0%, transparent 55%),
                linear-gradient(180deg, #0c1040 0%, #150d3a 60%, #1c0f45 100%);
    border: 1px solid #222a5c;
}

.moon { position:absolute; right:9%; top:26px; width:56px; height:56px; border-radius:50%;
        background: radial-gradient(circle at 35% 35%, #fff7d6, #f5d78a); box-shadow: 0 0 60px 16px rgba(245,215,138,.28); }
.skyline { position:absolute; left:0; bottom:0; width:100%; height:62%; }
.win { animation: tw 4s ease-in-out infinite; }
@keyframes tw { 0%,100% {opacity:1} 50% {opacity:.25} }
.hero-text {
    position: relative;
    z-index: 2;
    width: 100%;
    max-width: 650px;
    box-sizing: border-box;
    padding: 34px 32px 64px 32px;
}

.hero-kicker {
    color: #22d3ee;
    font-size: .74rem;
    letter-spacing: .22em;
    font-weight: 600;
    line-height: 1.45;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 7vw, 3.6rem);
    color: #fff;
    line-height: 1.05;
    margin: .25rem 0 .4rem;
    white-space: nowrap;
}

.hero-title span {
    background: linear-gradient(90deg,#22d3ee,#f472b6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-sub {
    color: #b9c0ee;
    font-size: 1rem;
    line-height: 1.5;
    max-width: 520px;
}

/* tabs */
.tab-nav button, button[role="tab"] { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:#8b93c4 !important; }
button[role="tab"][aria-selected="true"] { color:#22d3ee !important; border-color:#22d3ee !important; }

/* cards */
.card { background:#0f1330; border:1px solid #222a5c; border-radius:18px; padding:16px 18px !important; }
.card .card { border:none !important; padding:0 !important; background:transparent !important; box-shadow:none !important; }
.card-title { font-family:'Syne',sans-serif; font-weight:700; letter-spacing:.04em; font-size:.95rem; margin-bottom:4px; }
.card-title.cl { color:#22d3ee; } .card-title.hh { color:#f472b6; } .card-title.ot { color:#8b93c4; }

/* pill radios */
.pills .wrap { display:flex; gap:6px; flex-wrap:nowrap; background:#0b0f26; padding:4px; border-radius:14px; border:1px solid #2a3270; }
.pills label { flex:1; justify-content:center; text-align:center; border-radius:10px !important; border:none !important;
               background:transparent !important; padding:7px 0 !important; cursor:pointer; transition:.15s; color:#8b93c4 !important; }
.pills label.selected { background: linear-gradient(135deg,#f472b6,#a855f7) !important; color:#fff !important; box-shadow:0 4px 14px rgba(244,114,182,.35); }
.pills label span { color:inherit !important; font-weight:600; }
.pills input { display:none !important; }

/* preset buttons */
.preset { background:#0f1330 !important; border:1px solid #2a3270 !important; color:#c9d1f5 !important; border-radius:12px !important; font-weight:500 !important; }
.preset:hover { border-color:#22d3ee !important; color:#22d3ee !important; }

/* result */
.res { background: linear-gradient(160deg,#101642,#0d1130); border:1px solid #2a3270; border-radius:18px; padding:16px 18px; }
.g-num { fill:#fff; font-family:'Syne',sans-serif; font-weight:800; font-size:36px; }
.g-unit { fill:#8b93c4; font-size:11px; letter-spacing:.12em; text-transform:uppercase; }
.res-gauge { text-align:center; }
.badge { display:inline-block; margin-top:-2px; padding:3px 14px; border:1px solid; border-radius:999px; font-size:.8rem; font-weight:600; }
.stat-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin:14px 0; }
.stat-grid.four { grid-template-columns:repeat(4,1fr); }
.stat { background:#0b0f26; border:1px solid #222a5c; border-radius:12px; padding:10px 12px; }
.stat-k { color:#8b93c4; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
.stat-v { color:#fff; font-family:'Syne',sans-serif; font-weight:700; font-size:1.15rem; margin-top:2px; }
.drv-title { color:#c9d1f5; font-weight:600; font-size:.88rem; margin:4px 0 8px; }
.drv-title span { color:#6b7394; font-weight:400; font-size:.74rem; }
.drv { display:grid; grid-template-columns:130px 1fr 44px; align-items:center; gap:10px; margin:5px 0; }
.drv-name { color:#aab2e0; font-size:.82rem; }
.drv-track { background:#161c4a; height:9px; border-radius:6px; overflow:hidden; }
.drv-fill { height:100%; border-radius:6px; }
.drv-val { font-family:'Syne',sans-serif; font-size:.82rem; font-weight:700; text-align:right; }

/* notes */
.note { background:#0b0f26; border:1px solid #222a5c; border-radius:14px; padding:12px 16px; color:#aab2e0; font-size:.88rem; line-height:1.55; }
.note.tight { padding:8px 14px; margin-top:6px; }
.note b { color:#e6e9ff; } .note p { margin:8px 0 0; } .note code { color:#22d3ee; background:#0f1330; padding:1px 5px; border-radius:5px; }
.cvrow { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed #222a5c; }
.cvrow b { color:#22d3ee; }

@media (max-width: 720px) {

    .gradio-container {
        max-width: 100% !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .hero {
        min-height: 235px;
        border-radius: 17px;
    }

    .hero-text {
        max-width: 100%;
        padding: 24px 18px 92px 18px;
    }

    .hero-kicker {
        max-width: 72%;
        font-size: .58rem;
        letter-spacing: .13em;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    .hero-title {
        font-size: clamp(2.05rem, 12vw, 3rem);
        white-space: normal;
        line-height: .98;
        margin-top: .45rem;
    }

    .hero-sub {
        max-width: 82%;
        font-size: .88rem;
        line-height: 1.45;
    }

    .moon {
        width: 38px;
        height: 38px;
        top: 14px;
        right: 5%;
    }

    .skyline {
        height: 52%;
    }

    .stat-grid.four {
        grid-template-columns: repeat(2,1fr);
    }

    .drv {
        grid-template-columns: 104px 1fr 40px;
    }

    .tab-nav {
        overflow-x: auto !important;
        scrollbar-width: none;
    }

    .tab-nav::-webkit-scrollbar {
        display: none;
    }
}

@media (max-width: 430px) {

    .hero {
        min-height: 245px;
    }

    .hero-text {
        padding: 22px 16px 100px 16px;
    }

    .hero-kicker {
        max-width: 68%;
        font-size: .52rem;
        letter-spacing: .10em;
    }

    .hero-title {
        font-size: 1.85rem;
        letter-spacing: -0.03em;
        white-space: nowrap;
        overflow: visible;
    }

    .hero-sub {
        max-width: 88%;
        font-size: .82rem;
    }

    .stat-grid,
    .stat-grid.four {
        grid-template-columns: 1fr;
    }

    .drv {
        grid-template-columns: 92px 1fr 38px;
        gap: 7px;
    }

    .pills .wrap {
        flex-wrap: wrap;
    }

    .pills label {
        min-width: 38px;
    }
}
"""

THEME = gr.themes.Base(primary_hue="cyan", secondary_hue="pink", neutral_hue="slate")
FORCE_DARK = "() => { document.body.classList.add('dark'); }"


# ------------------------------------------------------------------ UI
def num(name, lo, hi, val, step=1, label=None):
    return gr.Slider(lo, hi, value=val, step=step, label=label or LABELS[name])


with gr.Blocks(title="WattWise · Household Energy Forecaster") as demo:
    gr.HTML(skyline_html())

    with gr.Tabs():
        # ------------------------------------------------ forecast
        with gr.Tab("⚡ Forecast"):
            with gr.Row():
                p1 = gr.Button("🏢 Studio · mild spring", elem_classes="preset")
                p2 = gr.Button("🏡 Family home · heatwave", elem_classes="preset")
                p3 = gr.Button("🏘️ Large house · cold snap", elem_classes="preset")

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title cl">☀️ CLIMATE</div>')
                        temp = num("Outdoor_Temp_C", 2, 42, 30, 0.5, "Outdoor temperature (°C)")
                        hum = num("Humidity_Pct", 15, 98, 60, 1, "Humidity (%)")
                        ac = num("AC_Hours", 0, 12, 5, 0.5, "AC running (hours/day)")
                        heat = num("Heating_Hours", 0, 8, 0, 0.5, "Heating running (hours/day)")
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title hh">🏠 HOUSEHOLD</div>')
                        occ = gr.Radio(["1", "2", "3", "4", "5", "6", "7"], value="4", label="Occupants",
                                       elem_classes="pills")
                        area = num("Home_Area_sqft", 350, 4200, 1800, 50, "Home area (sq ft)")
                        rooms = num("Num_Rooms", 1, 10, 4, 1, "Rooms")
                        appl = num("Appliance_Count", 3, 30, 13, 1, "Appliances")
                    with gr.Accordion("Other details (the model barely uses these)", open=False, elem_classes="card"):
                        noise = num("Neighbourhood_Noise_Index", 20, 90, 55, 1, "Neighbourhood noise index")
                        window = num("Window_Color_Code", 1, 5, 3, 1, "Window colour code")
                        pets = num("Pet_Count", 0, 5, 1, 1, "Pets")
                        roof = num("Roof_Age_Years", 1, 40, 20, 1, "Roof age (years)")
                        solar = num("Solar_Panel_Score", 0, 10, 5, 0.1, "Solar panel score")
                    with gr.Group(elem_classes="card"):
                        tariff = gr.Slider(3, 15, value=8, step=0.5, label="Electricity tariff (₹ per kWh, used for the bill estimate)")

                with gr.Column(scale=4):
                    result = gr.HTML()

            feature_inputs = {
                "Outdoor_Temp_C": temp, "Humidity_Pct": hum, "AC_Hours": ac, "Heating_Hours": heat,
                "Occupants": occ, "Home_Area_sqft": area, "Num_Rooms": rooms, "Appliance_Count": appl,
                "Neighbourhood_Noise_Index": noise, "Window_Color_Code": window, "Pet_Count": pets,
                "Roof_Age_Years": roof, "Solar_Panel_Score": solar,
            }
            all_inputs = [feature_inputs[f] for f in FEATURES] + [tariff]

            gr.on(triggers=[c.change for c in all_inputs], fn=forecast, inputs=all_inputs, outputs=result,
                  show_progress="hidden")
            demo.load(forecast, inputs=all_inputs, outputs=result)

            preset_targets = [temp, hum, ac, heat, occ, area, rooms, appl]
            p1.click(lambda: (24, 55, 1, 0, "1", 500, 1, 6), outputs=preset_targets)
            p2.click(lambda: (40, 35, 10, 0, "5", 2400, 5, 18), outputs=preset_targets)
            p3.click(lambda: (4, 85, 0, 6, "6", 3200, 8, 22), outputs=preset_targets)

        # ------------------------------------------------ insights
        with gr.Tab("🧠 Model insights"):
            gr.HTML(metrics_html())
            gr.HTML('<div class="card-title ot" style="margin:14px 2px 2px">🎚️ Drag alpha and watch the model refit</div>')
            alpha_exp = gr.Slider(-3, 0.6, value=round(float(np.log10(METRICS["alpha"])), 2), step=0.01,
                                  label="log10(alpha), the regularisation strength (right = stronger)")
            plot = gr.Plot(label="Elastic Net coefficients")
            plot_note = gr.HTML()
            alpha_exp.change(coef_plot, alpha_exp, [plot, plot_note], show_progress="hidden")
            demo.load(coef_plot, alpha_exp, [plot, plot_note])

        # ------------------------------------------------ about
        with gr.Tab("ℹ️ About the data"):
            gr.Markdown(
                f"""
### A synthetic dataset built to show off Elastic Net
`{len(DF)}` households, `{len(FEATURES)}` features, target = **Daily_Energy_kWh**. The data is *generated* (see
`generate_dataset.py`), so the true drivers are known and the model can be checked against them.

| Group | Features | Role |
|---|---|---|
| ☀️ Climate | outdoor temperature, humidity, AC hours, heating hours | correlated with each other, real effect |
| 🏠 Household | occupants, home area, rooms, appliances | strongly correlated with each other, real effect |
| 🎲 Unrelated | neighbourhood noise, window colour, pets, roof age, solar score | random, no real effect |

**Why Elastic Net?** Lasso (L1) tends to keep one feature from a correlated group and drop the others
arbitrarily. Ridge (L2) keeps all of them but never removes anything. Elastic Net blends both penalties: it can
shrink weak features while letting correlated ones share the credit.

**Honest caveats**
- The dataset is synthetic, so treat the numbers as a demonstration of the method, not a real-world energy model.
- Bill and CO₂ figures are simple assumptions (your tariff slider, ~{CO2_KG_PER_KWH} kg CO₂ per kWh).
- On this data Elastic Net and Lasso predict equally well. The difference is in *how* the weights are distributed.
"""
            )

if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CSS,
        js=FORCE_DARK,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )