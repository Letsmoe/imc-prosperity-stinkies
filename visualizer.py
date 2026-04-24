import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/2")
OUT_FILE = Path("market_data.png")

# Timestamp span per day (timestamps run 0–999_900 within a day)
DAY_SPAN = 1_000_000

# ── Auto-discover rounds, days and files ──────────────────────────────────────

def discover_rounds(data_dir: Path) -> dict:
    """
    Scan data/<round>/ folders and collect all matching price/trade CSVs.

    Returns
    -------
    {
        round_num (int): {
            "prices": {day (int): Path, ...},
            "trades": {day (int): Path, ...},
        },
        ...
    }
    """
    price_re = re.compile(r"prices_round_(\d+)_day_([-\d]+)\.csv$")
    trade_re = re.compile(r"trades_round_(\d+)_day_([-\d]+)\.csv$")

    rounds: dict = {}

    search_dirs = [p for p in data_dir.iterdir() if p.is_dir()] if data_dir.exists() else []
    search_dirs.append(data_dir)

    for folder in search_dirs:
        for csv in sorted(folder.glob("*.csv")):
            for pattern, kind in [(price_re, "prices"), (trade_re, "trades")]:
                m = pattern.match(csv.name)
                if m:
                    rnum = int(m.group(1))
                    day  = int(m.group(2))
                    rounds.setdefault(rnum, {"prices": {}, "trades": {}})
                    rounds[rnum][kind][day] = csv

    if not rounds:
        raise FileNotFoundError(
            f"No price/trade CSVs found under '{data_dir}'. "
            "Expected filenames like prices_round_1_day_-1.csv"
        )
    return rounds


# ── Load data for one round ───────────────────────────────────────────────────

def load_round(rdata: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Load prices and trades for a single round.
    Days are sorted chronologically and each gets a contiguous global_ts offset.
    Returns (prices_df, trades_df, sorted_days).
    """
    days = sorted(rdata["prices"].keys())

    price_frames, trade_frames = [], []

    for rank, day in enumerate(days):
        offset = rank * DAY_SPAN

        # --- prices ---
        if day in rdata["prices"]:
            df = pd.read_csv(rdata["prices"][day], sep=";")
            df["day"]       = day
            df["global_ts"] = df["timestamp"] + offset

            # FIX: compute mid_price if not present or all-zero/NaN
            if "mid_price" not in df.columns or df["mid_price"].eq(0).all() or df["mid_price"].isna().all():
                df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2

            # Replace any remaining 0s (rows where both sides were 0) with NaN
            # so matplotlib doesn't draw a spike down to 0
            df["mid_price"] = df["mid_price"].replace(0, np.nan)

            if "spread" not in df.columns:
                df["spread"] = df["ask_price_1"] - df["bid_price_1"]

            price_frames.append(df)

        # --- trades ---
        if day in rdata["trades"]:
            df = pd.read_csv(rdata["trades"][day], sep=";")
            df["day"]       = day
            df["global_ts"] = df["timestamp"] + offset
            trade_frames.append(df)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return prices, trades, days


# ── Plotting helpers ──────────────────────────────────────────────────────────

TITLE_KW  = dict(color="white", fontsize=11, fontweight="bold", pad=6)
LABEL_KW  = dict(color="#aaaaaa", fontsize=9)
TICK_KW   = dict(colors="#888888", labelsize=8)
GRID_KW   = dict(color="#2a2a3a", linewidth=0.5)
SPINE_CLR = "#2a2a3a"

_PALETTE = [
    "#e07b54", "#4caf85", "#7b8de0", "#e0c44c",
    "#c47be0", "#4cc0e0", "#e04c7b", "#85e04c",
]

def product_colors(products: list[str]) -> dict:
    return {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(products)}


def style_ax(ax, title=""):
    ax.set_facecolor("#1a1b26")
    ax.set_title(title, **TITLE_KW)
    ax.tick_params(axis="both", **TICK_KW)
    ax.xaxis.label.set(**LABEL_KW)
    ax.yaxis.label.set(**LABEL_KW)
    ax.grid(True, **GRID_KW)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)


def make_ts_formatter(days: list[int]) -> mticker.FuncFormatter:
    """Return an x-axis formatter that shows 'D{day} {ts/1000}k'."""
    def fmt(x, _):
        rank = int(x // DAY_SPAN)
        ts   = x % DAY_SPAN
        day_label = days[rank] if rank < len(days) else "?"
        if ts >= 1000:
            return f"D{day_label} {int(ts/1000)}k"
        return f"D{day_label} {int(ts)}"
    return mticker.FuncFormatter(fmt)


def add_day_dividers(ax, days: list[int]):
    """Draw vertical dashed lines between days and label each day band."""
    ymin, ymax = ax.get_ylim()
    n = len(days)
    for rank in range(1, n):
        boundary = rank * DAY_SPAN
        ax.axvline(boundary, color="#555577", linewidth=1.0, linestyle="--", zorder=2)
    for rank, day in enumerate(days):
        mid = rank * DAY_SPAN + DAY_SPAN / 2
        ax.text(mid, ymin + (ymax - ymin) * 0.02, f"Day {day}",
                color="#777799", fontsize=7, ha="center", va="bottom")


def apply_ts_axis(ax, days: list[int], n_ticks: int = 6):
    locator_step = max(DAY_SPAN // n_ticks, 100_000)
    ax.xaxis.set_major_formatter(make_ts_formatter(days))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(locator_step))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    add_day_dividers(ax, days)


# ── Per-product figure ────────────────────────────────────────────────────────

def plot_product(round_num: int, product: str, color: str,
                 prices: pd.DataFrame, trades: pd.DataFrame,
                 days: list[int], out_path: Path):
    """
    Save a single PNG for one product with 4 panels:
      Row 0: Mid price
      Row 1: Bid-ask spread
      Row 2: Trade prices scatter
      Row 3: Trade count & avg size bars (side by side)
    """
    fig = plt.figure(figsize=(14, 18))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.30,
                           top=0.93, bottom=0.06,
                           left=0.07, right=0.97)

    # ── Row 0: Mid price ─────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[0, :])
    sub = prices[prices["product"] == product].sort_values("global_ts")
    ax_mid.plot(sub["global_ts"], sub["mid_price"], color=color, linewidth=1.2)
    style_ax(ax_mid, f"{product} – Mid Price")
    ax_mid.set_xlabel("Timestamp")
    ax_mid.set_ylabel("Price")
    apply_ts_axis(ax_mid, days)

    # ── Row 1: Spread ────────────────────────────────────────────────────────
    ax_spread = fig.add_subplot(gs[1, :])
    ax_spread.plot(sub["global_ts"], sub["spread"], color=color, linewidth=1.2)
    style_ax(ax_spread, f"{product} – Bid-Ask Spread")
    ax_spread.set_xlabel("Timestamp")
    ax_spread.set_ylabel("Spread")
    apply_ts_axis(ax_spread, days)

    # ── Row 2: Trade scatter ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2, :])
    if not trades.empty:
        tsub = trades[trades["symbol"] == product].sort_values("global_ts")
        if not tsub.empty:
            max_q = tsub["quantity"].max()
            sizes = (tsub["quantity"] / max_q * 80).clip(lower=6)
            ax_t.scatter(tsub["global_ts"], tsub["price"],
                         s=sizes, color=color, alpha=0.65, linewidths=0)
    style_ax(ax_t, f"{product} – Trade Prices  (bubble ∝ quantity)")
    ax_t.set_xlabel("Timestamp")
    ax_t.set_ylabel("Trade Price")
    apply_ts_axis(ax_t, days)

    # ── Row 3: Trade count & avg size bars ───────────────────────────────────
    ax_cnt = fig.add_subplot(gs[3, 0])
    ax_avg = fig.add_subplot(gs[3, 1])

    bar_w      = 0.7 / max(len(days), 1)
    x          = np.array([0])
    day_alphas = np.linspace(0.4, 1.0, len(days))

    for i, day in enumerate(days):
        sub_d  = trades[(trades["day"] == day) & (trades["symbol"] == product)] \
                 if not trades.empty else pd.DataFrame()
        cnt  = len(sub_d)
        avg  = sub_d["quantity"].mean() if not sub_d.empty else 0
        offset = (i - (len(days) - 1) / 2) * bar_w
        alpha  = float(day_alphas[i])
        ax_cnt.bar(x + offset, cnt, bar_w * 0.92, color=color, alpha=alpha, label=f"Day {day}")
        ax_avg.bar(x + offset, avg, bar_w * 0.92, color=color, alpha=alpha, label=f"Day {day}")

    for ax, title, ylabel in [
        (ax_cnt, "Trade Count by Day",        "Number of Trades"),
        (ax_avg, "Average Trade Size by Day", "Avg Quantity"),
    ]:
        style_ax(ax, title)
        ax.set_ylabel(ylabel, **LABEL_KW)
        ax.set_xticks([0])
        ax.set_xticklabels([product], color="white", fontsize=9)
        ax.legend(fontsize=8, facecolor="#1a1b26", labelcolor="white", edgecolor=SPINE_CLR)

    title_days = "  ·  ".join(f"Day {d}" for d in days)
    fig.suptitle(f"IMC Prosperity  ·  Round {round_num}  ·  {product}  ·  {title_days}",
                 color="white", fontsize=14, fontweight="bold", y=0.97)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  {product} → saved to {out_path.resolve()}")
    plt.close(fig)


# ── Per-round dispatcher ──────────────────────────────────────────────────────

def plot_round(round_num: int, prices: pd.DataFrame, trades: pd.DataFrame,
               days: list[int], out_dir: Path):

    products = sorted(prices["product"].unique()) if not prices.empty else []
    colors   = product_colors(products)

    if not products:
        print(f"Round {round_num}: no products found, skipping.")
        return

    for product in products:
        # Sanitise product name for use in filename
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", product)
        out_path  = out_dir / f"market_data_round_{round_num}_{safe_name}.png"
        plot_product(round_num, product, colors[product],
                     prices, trades, days, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rounds = discover_rounds(DATA_DIR)
    print(f"Discovered rounds: {sorted(all_rounds.keys())}")

    out_dir = OUT_FILE.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for round_num in sorted(all_rounds.keys()):
        rdata = all_rounds[round_num]
        print(f"\nLoading round {round_num}  "
              f"(price days: {sorted(rdata['prices'].keys())}, "
              f"trade days: {sorted(rdata['trades'].keys())})")

        prices, trades, days = load_round(rdata)
        plot_round(round_num, prices, trades, days, out_dir)


if __name__ == "__main__":
    main()