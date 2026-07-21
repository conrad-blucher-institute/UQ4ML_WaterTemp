"""Generate the deck's diagram PNGs (the non-data-chart visuals).

Writes to docs/deck_assets/:
  funnel_phases.png     — slide 3: phased-tuning funnel (replaces mermaid screenshot)
  funnel_phases_v2.png  — slide 3b: decluttered layout variant (pick v1 or v2)
  sharding_map.png      — slide 6: 40-shard -> lead-time map (replaces mermaid)
  leadtime_ladder.png   — slide 7: 21-lead-time ladder, core 4 emphasized
  patience_fraction.png — slide 9: patience-25 as a fraction of training length

Data-driven charts (stability/seed-noise) come from scripts/tuning_tests/;
this script is only for diagrams whose content is fixed by the campaign design.
Run in the NLL_streamlit env:  python scripts/make_deck_visuals.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "deck_assets"

# semantic colors, matching the deck's mermaid legend:
# blue = compute phases / grid, green = selection & deliverable, ink for text
BLUE, GREEN, INK, MUTED = "#3d6fa8", "#2e8b6d", "#2b2b2b", "#8a8a8a"

plt.rcParams.update({
    "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
    "figure.facecolor": "white",
})


def _box(ax, xy, w, h, text, edge, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                fill=False, edgecolor=edge, linewidth=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=11, fontweight="bold" if bold else "normal", color=INK)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                                 color=MUTED, linewidth=1.5))


def funnel_phases():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _box(ax, (1.0, 8.0), 8.0, 1.5,
         "Phase 1 — SCREEN\nfull 2,880-config grid • 3 repetitions each", BLUE, True)
    _box(ax, (2.5, 5.4), 5.0, 1.4,
         "Phase 2 — CONFIRM\ntop ~20 configs per lead time • 10+ reps", BLUE, True)
    _box(ax, (3.4, 2.8), 3.2, 1.4,
         "WINNER per lead time\nbest mean val + low std\n(2021/Uri = veto only)", GREEN)
    _box(ax, (3.4, 0.3), 3.2, 1.3,
         "Deliverable: the rep closest to\nthe config mean (typical, not lucky)", GREEN)
    _arrow(ax, (5, 8.0), (5, 6.9))
    _arrow(ax, (5, 5.4), (5, 4.3))
    _arrow(ax, (5, 2.8), (5, 1.7))
    ax.text(8.6, 6.35, "rank by mean val metric\n(low std as tiebreak)",
            fontsize=9, color=MUTED, ha="left", va="center")
    ax.text(9.1, 8.75, "blue = compute\ngreen = selection", fontsize=8,
            color=MUTED, ha="right", va="center",
            bbox=dict(boxstyle="round", fill=False, edgecolor=MUTED, linewidth=0.8))
    fig.savefig(OUT / "funnel_phases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def funnel_phases_v2():
    """Same funnel, decluttered: side annotation and legend get their own
    clear space so nothing overlaps the boxes. Slide 3b; pick v1 or v2."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    _box(ax, (0.6, 8.0), 7.4, 1.5,
         "Phase 1 — SCREEN\nfull 2,880-config grid • 3 repetitions each", BLUE, True)
    _box(ax, (1.8, 5.4), 5.0, 1.4,
         "Phase 2 — CONFIRM\ntop ~20 configs per lead time • 10+ reps", BLUE, True)
    _box(ax, (2.7, 2.8), 3.2, 1.4,
         "WINNER per lead time\nbest mean val + low std\n(2021/Uri = veto only)", GREEN)
    _box(ax, (2.7, 0.3), 3.2, 1.3,
         "Deliverable: the rep closest to\nthe config mean (typical, not lucky)", GREEN)
    _arrow(ax, (4.3, 8.0), (4.3, 6.9))
    _arrow(ax, (4.3, 5.4), (4.3, 4.3))
    _arrow(ax, (4.3, 2.8), (4.3, 1.7))
    ax.text(8.6, 6.1, "rank by mean val metric\n(low std as tiebreak)",
            fontsize=9, color=MUTED, ha="left", va="center")
    ax.text(11.8, 0.9, "blue = compute\ngreen = selection", fontsize=8,
            color=MUTED, ha="right", va="center",
            bbox=dict(boxstyle="round", fill=False, edgecolor=MUTED, linewidth=0.8))
    fig.savefig(OUT / "funnel_phases_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sharding_map():
    lts = [("LT 12 h", 1), ("LT 48 h", 11), ("LT 96 h", 21), ("LT 120 h", 31)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-2.4, 10.6); ax.set_ylim(-1.4, 4.6); ax.axis("off")
    for row, (label, first) in enumerate(reversed(lts)):
        y = row * 1.1
        ax.text(-0.4, y + 0.35, label, ha="right", va="center",
                fontweight="bold", fontsize=11)
        for i in range(10):
            ax.add_patch(FancyBboxPatch((i, y), 0.85, 0.7,
                                        boxstyle="round,pad=0.01",
                                        fill=False, edgecolor=BLUE, linewidth=1.6))
            ax.text(i + 0.425, y + 0.35, str(first + i), ha="center",
                    va="center", fontsize=9)
    ax.text(5, -0.9, "40 shards × 72 contiguous jobs — lead time is the grid's "
            "outermost loop, so shards align with lead times\nper-shard CSV "
            "checkpointing: crash-safe resume, no shared mutable files",
            ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(5, 4.35, "esb run --profile mape_scaled --shard k/40",
            ha="center", family="monospace", fontsize=10.5)
    fig.savefig(OUT / "sharding_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# 17-LT extension status — update these two lines when shards finish, then
# rerun this script. Source: completed-row counts in
# results/esb_tuner_scaled_17lt/mape_progress_shard*of17.csv (720x3 = done).
EXT_DONE = {3, 6, 18, 24, 30, 36, 42, 54}      # as of 2026-07-16
EXT_RUNNING = {60, 114}


def leadtime_ladder():
    all_lt = [3] + list(range(6, 121, 6))
    core = {12, 48, 96, 120}
    done = core | EXT_DONE
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.set_xlim(-4, 127); ax.set_ylim(-1.6, 2.4); ax.axis("off")
    ax.hlines(0, 0, 123, color=MUTED, linewidth=1)
    for lt in all_lt:
        is_core = lt in core
        if lt in done:
            color = BLUE
        elif lt in EXT_RUNNING:
            color = GREEN
        else:
            color = MUTED
        ax.vlines(lt, 0, 0.55 if lt in done else 0.3, color=color,
                  linewidth=2.5 if lt in done else 1.2)
        ax.text(lt, 0.75 if is_core else -0.55, str(lt), ha="center",
                fontsize=9.5 if is_core else 7.5,
                fontweight="bold" if is_core else "normal", color=color)
    ax.text(0, 2.0, "21 lead times (hours) — blue = tuned (bold = core 4), "
            f"green = running now, grey = queued "
            f"({len(done)}/21 done)", fontsize=10, ha="left")
    fig.savefig(OUT / "leadtime_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def patience_fraction():
    epochs = list(range(100, 2501, 25))
    frac = [25 / e * 100 for e in epochs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, frac, color=BLUE, linewidth=2)
    for e, note in [(107, "shortest observed run:\npatience = 23% of training"),
                    (2500, "longest runs:\npatience = 1%")]:
        f = 25 / e * 100
        ax.plot([e], [f], "o", color=BLUE, markersize=8)
        ax.annotate(note, (e, f), textcoords="offset points",
                    xytext=(14, 18) if e < 300 else (-10, 22),
                    ha="left" if e < 300 else "right", fontsize=9, color=INK)
    ax.set_xlabel("total training length (epochs)")
    ax.set_ylabel("patience 25 as % of training")
    ax.set_title("One patience value is a different rule for every run",
                 fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "patience_fraction.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (funnel_phases, funnel_phases_v2, sharding_map, leadtime_ladder,
               patience_fraction):
        fn()
        print(f"wrote {OUT / (fn.__name__ + '.png')}")


if __name__ == "__main__":
    main()
