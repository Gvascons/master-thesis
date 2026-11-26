"""Short-dissertation variant of the 14-model decision flowchart.

Differences from figures_ai1_display.fig_flowchart14 (kept fixed there for
the long dissertation and the AI-1 paper): the distillation cross-reference
is "Ch. 6"; box 2 drops "context cap" (a context overflow is handled by
subsampling, not a structural exclusion — B13/D1); and the "yes" arrows
start at the decision boxes' left edge instead of a free point (B13/D9).
Output: results/figures/decision_flowchart_14_en_short.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures_ai1_display import save


def main(ref="Ch. 6", out="decision_flowchart_14_en_short"):
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.axis("off")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.03, 1.03)

    def box(x, y, t, c="#eef"):
        ax.annotate(t, (x, y), ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.45", fc=c, ec="#333"))

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, lbl, fontsize=12,
                    color="#a00", ha="left",
                    bbox=dict(fc="white", ec="none", pad=1.0))

    box(0.58, 0.985, "Tabular supervised problem")
    box(0.58, 0.86, "1. Serving-latency budget critical?\n"
                    "(FMs cost 4–5 orders of magnitude more per row)")
    arrow(0.58, 0.955, 0.58, 0.905)
    box(0.17, 0.68, "GBDT family\n(XGBoost / LightGBM / CatBoost)\n"
                    "in distributional regression, distillation\n"
                    f"recovers part of the TabPFN edge at ~10\u00b2 \u00b5s/row ({ref})",
        "#fde")
    arrow(0.44, 0.835, 0.23, 0.745, "yes")
    box(0.58, 0.62, "2. Architectural limits hit?\n(>10 classes)")
    arrow(0.58, 0.815, 0.58, 0.665, "no")
    box(0.17, 0.46, "GBDT or DL\n(task-matched; see reversals)", "#fde")
    arrow(0.48, 0.60, 0.23, 0.50, "yes")
    box(0.58, 0.40, "3. Auditable explanations required?\n"
                    "(regulation; post-hoc attribution)")
    arrow(0.58, 0.575, 0.58, 0.445, "no")
    box(0.16, 0.24, "GBDT\n(attribution-explainable,\n"
                    "accepted in regulated practice)", "#fde")
    arrow(0.44, 0.38, 0.23, 0.295, "yes")
    box(0.54, 0.12, "4. No active constraint → FOUNDATION MODEL FIRST\n"
                    "TabPFN (mature) · TabFM (frontier, v1.0.1 caveats)\n"
                    "median normalized regret 0.000 @14 models (descriptive; executable:\n"
                    "always-TabFM 0.000 median / 0.080 mean; always-TabPFN 0.242 median)",
        "#efe")
    arrow(0.58, 0.355, 0.56, 0.195, "no")
    save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
