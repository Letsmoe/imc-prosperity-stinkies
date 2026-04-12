import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
OUT_FILE = Path("market_data.png")

PRICE_FILES = {
    -1: DATA_DIR / "prices_round_0_day_-1.csv",
    -2: DATA_DIR / "prices_round_0_day_-2.csv",
}
TRADE_FILES = {
    -1: DATA_DIR / "trades_round_0_day_-1.csv",
    -2: DATA_DIR / "trades_round_0_day_-2.csv",
}

PRODUCTS   = ["TOMATOES", "EMERALDS"]
DAYS       = [-2, -1]          # chronological order
COLORS     = {"TOMATOES": "#e07b54", "EMERALDS": "#4caf85"}
DAY_LABELS = {-2: "Day −2", -1: "Day −1"}

# Each day's timestamps run 0–999900; offset day -1 by 1_000_000 so they
# sit end-to-end on a single continuous axis.
DAY_OFFSET = {-2: 0, -1: 1_000_000}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    frames = []
    for day, path in PRICE_FILES.items():
        df = pd.read_csv(path, sep=";")
        df["global_ts"] = df["timestamp"] + DAY_OFFSET[day]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for day, path in TRADE_FILES.items():
        df = pd.read_csv(path, sep=";")
        df["day"] = -1 if "day_-1" in str(TRADE_FILES[day]) else -2
        df["global_ts"] = df["timestamp"] + DAY_OFFSET[df["day"].iloc[0]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


prices = load_prices()
trades = load_trades()

# Derived columns
prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]

# ── Shared x-axis formatter & day-boundary helpers ────────────────────────────

DAY_BOUNDARY = 1_000_000          # global_ts where day -2 ends / day -1 begins

def fmt_global_ts(x, _):
    """Show timestamp within its day, suffixed with k, plus day label."""
    if x < DAY_BOUNDARY:
        ts = x
        day_str = "D-2 "
    else:
        ts = x - DAY_BOUNDARY
        day_str = "D-1 "
    return f"{day_str}{int(ts/1000)}k" if ts >= 1000 else f"{day_str}{int(ts)}"

def add_day_divider(ax):
    """Draw a vertical dashed line at the day boundary and label each day."""
    ymin, ymax = ax.get_ylim()
    ax.axvline(DAY_BOUNDARY, color="#555577", linewidth=1.0, linestyle="--", zorder=2)
    # Small day labels just above the x-axis
    mid1 = DAY_BOUNDARY / 2
    mid2 = DAY_BOUNDARY + DAY_BOUNDARY / 2
    for mid, lbl in [(mid1, "Day −2"), (mid2, "Day −1")]:
        ax.text(mid, ymin + (ymax - ymin) * 0.02, lbl,
                color="#777799", fontsize=7, ha="center", va="bottom")

# ── Figure layout ─────────────────────────────────────────────────────────────
#
#  Row 0 │  Mid-price over time (TOMATOES)  │  Mid-price over time (EMERALDS)
#  Row 1 │  Bid-ask spread (TOMATOES)       │  Bid-ask spread (EMERALDS)
#  Row 2 │  Trade price & volume (TOMATOES) │  Trade price & volume (EMERALDS)
#  Row 3 │  Trade count per day (both)      │  Avg trade size per product/day

fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor("#0f1117")

TITLE_KW  = dict(color="white", fontsize=11, fontweight="bold", pad=6)
LABEL_KW  = dict(color="#aaaaaa", fontsize=9)
TICK_KW   = dict(colors="#888888", labelsize=8)
GRID_KW   = dict(color="#2a2a3a", linewidth=0.5)
SPINE_CLR = "#2a2a3a"

gs_main = gridspec.GridSpec(4, 2, figure=fig,
                             hspace=0.55, wspace=0.30,
                             top=0.93, bottom=0.06,
                             left=0.07, right=0.97)

def style_ax(ax, title=""):
    ax.set_facecolor("#1a1b26")
    ax.set_title(title, **TITLE_KW)
    ax.tick_params(axis="both", **TICK_KW)
    ax.xaxis.label.set(**LABEL_KW)
    ax.yaxis.label.set(**LABEL_KW)
    ax.grid(True, **GRID_KW)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)

# ── Row 0 & 1: Mid-price and spread per product (continuous x-axis) ───────────

for col, product in enumerate(PRODUCTS):
    ax_mid    = fig.add_subplot(gs_main[0, col])
    ax_spread = fig.add_subplot(gs_main[1, col])
    color = COLORS[product]

    sub = prices[prices["product"] == product].sort_values("global_ts")
    ax_mid.plot(sub["global_ts"], sub["mid_price"], color=color, linewidth=1.2)
    ax_spread.plot(sub["global_ts"], sub["spread"],  color=color, linewidth=1.2)

    style_ax(ax_mid,    f"{product} – Mid Price")
    style_ax(ax_spread, f"{product} – Bid-Ask Spread")
    ax_mid.set_xlabel("Timestamp")
    ax_mid.set_ylabel("Price")
    ax_spread.set_xlabel("Timestamp")
    ax_spread.set_ylabel("Spread")
    for ax in (ax_mid, ax_spread):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_global_ts))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(200_000))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        add_day_divider(ax)

# ── Row 2: Trade price (scatter) per product (continuous x-axis) ──────────────

for col, product in enumerate(PRODUCTS):
    ax_t = fig.add_subplot(gs_main[2, col])
    color = COLORS[product]

    sub = trades[trades["symbol"] == product].sort_values("global_ts")
    if not sub.empty:
        max_q = sub["quantity"].max()
        sizes = (sub["quantity"] / max_q * 80).clip(lower=6)
        ax_t.scatter(sub["global_ts"], sub["price"],
                     s=sizes, color=color, alpha=0.65, linewidths=0)

    style_ax(ax_t, f"{product} – Trade Prices  (bubble ∝ quantity)")
    ax_t.set_xlabel("Timestamp")
    ax_t.set_ylabel("Trade Price")
    ax_t.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_global_ts))
    ax_t.xaxis.set_major_locator(mticker.MultipleLocator(200_000))
    plt.setp(ax_t.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    add_day_divider(ax_t)

# ── Row 3: Trade count & avg size comparison bars ────────────────────────────

ax_cnt = fig.add_subplot(gs_main[3, 0])
ax_avg = fig.add_subplot(gs_main[3, 1])

bar_w = 0.35
x     = np.arange(len(PRODUCTS))

for i, day in enumerate(DAYS):
    sub  = trades[trades["day"] == day]
    cnts = [len(sub[sub["symbol"] == p]) for p in PRODUCTS]
    avgs = [sub[sub["symbol"] == p]["quantity"].mean() for p in PRODUCTS]
    offset = (i - 0.5) * bar_w
    bar_color = [COLORS[p] for p in PRODUCTS]
    alpha = 0.55 if day == -2 else 1.0

    for j, (cnt, avg, bc) in enumerate(zip(cnts, avgs, bar_color)):
        ax_cnt.bar(x[j] + offset, cnt, bar_w * 0.95, color=bc,
                   alpha=alpha, label=DAY_LABELS[day] if j == 0 else "")
        ax_avg.bar(x[j] + offset, avg, bar_w * 0.95, color=bc,
                   alpha=alpha, label=DAY_LABELS[day] if j == 0 else "")

for ax, title, ylabel in [
    (ax_cnt, "Trade Count by Product & Day", "Number of Trades"),
    (ax_avg, "Average Trade Size by Product & Day", "Avg Quantity"),
]:
    style_ax(ax, title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(PRODUCTS, color="white", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1a1b26", labelcolor="white",
              edgecolor=SPINE_CLR)

fig.suptitle("IMC Prosperity  ·  Round 0  ·  Market Data Overview",
             color="white", fontsize=14, fontweight="bold", y=0.97)

fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_FILE.resolve()}")
plt.show()