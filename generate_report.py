"""
generate_report.py

Assembles a 3-page PDF presentation from project visuals and findings.

Requires visuals/*.png files — run the analysis pipeline first:
    python eda_narrative.py
    python -m src.risk_model
    python -m src.payoff_model

Output: outputs/report.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

VISUALS_DIR = "visuals"
OUTPUT_DIR = "outputs"
GREEN = "#86BC25"
DARK = "#1A1A1A"
LIGHT = "#F7F7F7"
MID = "#CCCCCC"


def _img(filename):
    path = os.path.join(VISUALS_DIR, filename)
    return mpimg.imread(path) if os.path.exists(path) else None


def _place_image(fig, img, rect, placeholder_text="Run pipeline to generate"):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor(LIGHT)
        ax.text(0.5, 0.5, placeholder_text, ha="center", va="center",
                fontsize=9, color=MID, style="italic", transform=ax.transAxes)


def _header(fig, title, page_label):
    ax = fig.add_axes([0, 0.935, 1, 0.065])
    ax.set_facecolor(GREEN)
    ax.axis("off")
    ax.text(0.02, 0.5, title, color="white", fontsize=13, fontweight="bold",
            va="center", transform=ax.transAxes)
    ax.text(0.98, 0.5, f"Expected Value of Marriage Analysis  |  {page_label}",
            color="white", fontsize=8.5, va="center", ha="right",
            transform=ax.transAxes)


def _footer_box(fig, rows, rect):
    """Render a list of (label, body) bullet rows in a shaded box."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(LIGHT)
    ax.axis("off")
    y = 0.90
    for label, body in rows:
        ax.text(0.015, y, f"\u2022 {label}:", fontsize=9, fontweight="bold",
                color=GREEN, va="top", transform=ax.transAxes)
        ax.text(0.17, y, body, fontsize=9, color=DARK, va="top",
                transform=ax.transAxes, wrap=True)
        y -= 0.22


# -----------------------------------------------------------------------
# Page 1 — Model Performance
# -----------------------------------------------------------------------

def _page_model_performance(pdf):
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    _header(fig, "Model Performance", "1 / 3")

    _place_image(fig, _img("logit_divorce_roc.png"),         [0.03, 0.34, 0.44, 0.57])
    _place_image(fig, _img("logit_feature_importance.png"),  [0.52, 0.34, 0.45, 0.57])

    # Metrics table
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.29])
    ax.set_facecolor(LIGHT)
    ax.axis("off")

    headers = ["Metric", "Value", "Interpretation"]
    rows = [
        ("ROC AUC",          "0.687",              "Acceptable discrimination (0.5 = random)"),
        ("Brier Score",      "0.093",              "Good probabilistic calibration"),
        ("5-Fold CV AUC",    "0.687 \u00b1 0.001", "Highly stable — no overfitting detected"),
        ("vs. Random Forest","< 0.01 AUC gap",     "Logistic retained: interpretable log-odds coefficients, better calibration"),
    ]

    col_x = [0.01, 0.18, 0.42]
    y = 0.88
    for j, h in enumerate(headers):
        ax.text(col_x[j], y, h, fontsize=9.5, fontweight="bold", color=GREEN,
                va="top", transform=ax.transAxes)
    ax.plot([0.01, 0.99], [0.72, 0.72], color=GREEN, linewidth=1,
            transform=ax.transAxes, clip_on=False)
    y = 0.65
    for label, value, interp in rows:
        ax.text(col_x[0], y, label,  fontsize=9,   color=DARK, va="top", transform=ax.transAxes)
        ax.text(col_x[1], y, value,  fontsize=9,   color=DARK, va="top", transform=ax.transAxes)
        ax.text(col_x[2], y, interp, fontsize=8.5, color=DARK, va="top", transform=ax.transAxes)
        y -= 0.19

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Page 2 — Key Findings
# -----------------------------------------------------------------------

def _page_key_findings(pdf):
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    _header(fig, "Key Findings — Divorce Risk and Economic Effects", "2 / 3")

    _place_image(fig, _img("divorce_rate_by_education.png"), [0.03, 0.34, 0.44, 0.57])
    _place_image(fig, _img("divorce_rate_by_age_sex.png"),   [0.52, 0.34, 0.45, 0.57])

    _footer_box(fig, [
        ("Education",            "Strongest predictor. College-educated: 3\u20136% divorce rate. HS only: 6\u201312%."),
        ("Income effect",        "Married premium over never-married: +25\u201337% (regression-controlled). Divorced penalty vs. married: \u221211\u201324%."),
        ("Sex",                  "Women have slightly higher observed divorce rates at equivalent age and education."),
        ("Marriage cohort",      "Recent marriages show lower cross-sectional rates \u2014 duration artifact, not a declining-divorce-rate trend."),
    ], [0.03, 0.02, 0.94, 0.29])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Page 3 — EV Analysis
# -----------------------------------------------------------------------

def _page_ev_analysis(pdf):
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    _header(fig, "Expected Value Analysis", "3 / 3")

    sensitivity = _img("sensitivity_analysis.png")
    if sensitivity is None:
        sensitivity = _img("prob_divorced_given_yrmarr_sex.png")
    _place_image(fig, sensitivity, [0.03, 0.34, 0.57, 0.57],
                 placeholder_text="Run: python -m src.payoff_model")

    # EV results panel
    ax = fig.add_axes([0.63, 0.34, 0.34, 0.57])
    ax.set_facecolor(LIGHT)
    ax.axis("off")
    ax.text(0.5, 0.95, "Sample EV Results", fontsize=11, fontweight="bold",
            ha="center", va="top", color=DARK, transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.82, 0.82], color=GREEN, linewidth=1,
            transform=ax.transAxes, clip_on=False)

    cols = [0.02, 0.60, 0.80]
    y = 0.76
    for h, c in zip(["Profile", "P(div)", "\u0394EV"], cols):
        ax.text(c, y, h, fontsize=8.5, fontweight="bold", color=GREEN,
                va="top", transform=ax.transAxes)
    y -= 0.12

    profiles = [
        ("Male, 32, college, $80K",      "~6%",  "+"),
        ("Female, 29, grad school, $120K","~5%",  "+"),
        ("Male, 25, HS only, $40K",       "~12%", "+"),
    ]
    for name, p, delta in profiles:
        ax.text(cols[0], y, name,  fontsize=8,   color=DARK,  va="top", transform=ax.transAxes)
        ax.text(cols[1], y, p,     fontsize=8,   color=DARK,  va="top", transform=ax.transAxes)
        ax.text(cols[2], y, delta, fontsize=9,   color=GREEN, va="top", fontweight="bold",
                transform=ax.transAxes)
        y -= 0.13

    ax.text(0.02, y - 0.04,
            "Break-even: marriage has positive\nEV until P(divorce) > ~57%.\nNo observed group approaches this.",
            fontsize=8.5, color=DARK, va="top", transform=ax.transAxes,
            linespacing=1.5)

    _footer_box(fig, [
        ("Sensitivity",  "Positive EV holds across horizons 5\u201330 yrs, discount rates 1\u20137%, divorce costs $0\u2013$50K."),
        ("Key caveat",   "Income effects may partly reflect selection. Full causal ID requires IV/DiD beyond cross-sectional ACS."),
        ("Recommended",  "Temporal validation (train earlier years, test later) and geographic disaggregation as next steps."),
    ], [0.03, 0.02, 0.94, 0.29])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "report.pdf")

    with PdfPages(out) as pdf:
        d = pdf.infodict()
        d["Title"]   = "Expected Value of Marriage Analysis"
        d["Author"]  = "Jack Milligan"
        d["Subject"] = "Divorce risk modeling and EV analysis — IPUMS ACS"

        _page_model_performance(pdf)
        _page_key_findings(pdf)
        _page_ev_analysis(pdf)

    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()