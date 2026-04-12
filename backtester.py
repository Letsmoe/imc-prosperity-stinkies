import importlib
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from datamodel import Order, OrderDepth, Trade, TradingState, Observation, Listing

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR      = Path("data")
OUT_FILE      = Path("backtest.png")
TRADER_MODULE = "ella"          # your trader filename without .py

PRICE_FILES = {
    -2: DATA_DIR / "prices_round_0_day_-2.csv",
    -1: DATA_DIR / "prices_round_0_day_-1.csv",
}
TRADE_FILES = {
    -2: DATA_DIR / "trades_round_0_day_-2.csv",
    -1: DATA_DIR / "trades_round_0_day_-1.csv",
}

POSITION_LIMIT = 80
PRODUCTS   = ["TOMATOES", "EMERALDS"]
DAYS       = [-2, -1]
COLORS     = {"TOMATOES": "#e07b54", "EMERALDS": "#4caf85"}
DAY_OFFSET = {-2: 0, -1: 1_000_000}

# ── Load CSV data ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    frames = []
    for day, path in PRICE_FILES.items():
        df = pd.read_csv(path, sep=";")
        df["global_ts"] = df["timestamp"] + DAY_OFFSET[day]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("global_ts")
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def load_market_trades() -> pd.DataFrame:
    frames = []
    for day, path in TRADE_FILES.items():
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        df["global_ts"] = df["timestamp"] + DAY_OFFSET[day]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("global_ts")


# ── Build OrderDepth from one price row ───────────────────────────────────────

def row_to_order_depth(row: pd.Series) -> OrderDepth:
    od = OrderDepth()
    for level in [1, 2, 3]:
        bp = row.get(f"bid_price_{level}")
        bv = row.get(f"bid_volume_{level}")
        ap = row.get(f"ask_price_{level}")
        av = row.get(f"ask_volume_{level}")
        if pd.notna(bp) and pd.notna(bv):
            od.buy_orders[int(bp)]  =  int(bv)
        if pd.notna(ap) and pd.notna(av):
            od.sell_orders[int(ap)] = -int(av)   # negative convention
    return od


# ── Order matching engine ─────────────────────────────────────────────────────

def match_orders(
    orders: List[Order],
    order_depth: OrderDepth,
    position: Dict[str, int],
    product: str,
) -> List[Trade]:
    fills = []
    pos   = position.get(product, 0)

    for order in orders:
        remaining = order.quantity

        if remaining > 0:   # BUY — lift the ask
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price > order.price:
                    break
                available = -order_depth.sell_orders[ask_price]
                can_buy   = min(remaining, available, POSITION_LIMIT - pos)
                if can_buy <= 0:
                    break
                fills.append(Trade(product, ask_price, can_buy,
                                   buyer="SUBMISSION", timestamp=0))
                pos       += can_buy
                remaining -= can_buy
                if remaining == 0:
                    break

        elif remaining < 0:  # SELL — hit the bid
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price < order.price:
                    break
                available = order_depth.buy_orders[bid_price]
                can_sell  = min(-remaining, available, POSITION_LIMIT + pos)
                if can_sell <= 0:
                    break
                fills.append(Trade(product, bid_price, -can_sell,
                                   seller="SUBMISSION", timestamp=0))
                pos       -= can_sell
                remaining += can_sell
                if remaining == 0:
                    break

    position[product] = pos
    return fills


# ── Backtester ────────────────────────────────────────────────────────────────

def run_backtest(trader) -> pd.DataFrame:
    prices_df     = load_prices()
    mkt_trades_df = load_market_trades()

    position:    Dict[str, int]   = defaultdict(int)
    cash:        Dict[str, float] = defaultdict(float)
    trader_data: str              = ""
    own_trades_prev               = defaultdict(list)

    records = []
    mkt_by_ts = mkt_trades_df.groupby("global_ts")

    for global_ts, tick_df in prices_df.groupby("global_ts"):
        timestamp = int(tick_df["timestamp"].iloc[0])

        order_depths: Dict[str, OrderDepth] = {}
        market_trades_tick = defaultdict(list)

        for _, row in tick_df.iterrows():
            order_depths[row["product"]] = row_to_order_depth(row)

        if global_ts in mkt_by_ts.groups:
            for _, tr in mkt_by_ts.get_group(global_ts).iterrows():
                market_trades_tick[tr["symbol"]].append(
                    Trade(tr["symbol"], int(tr["price"]), int(tr["quantity"]),
                          timestamp=int(tr["timestamp"]))
                )

        state = TradingState(
            traderData    = trader_data,
            timestamp     = timestamp,
            listings      = {p: {} for p in PRODUCTS},
            order_depths  = order_depths,
            own_trades    = dict(own_trades_prev),
            market_trades = dict(market_trades_tick),
            position      = dict(position),
            observations  = Observation({}, {}),
        )

        try:
            result, conversions, trader_data = trader.run(state)
        except Exception as e:
            print(f"[ts={global_ts}] Trader error: {e}")
            result, trader_data = {}, ""

        own_trades_prev = defaultdict(list)
        for product, orders in result.items():
            od    = order_depths.get(product, OrderDepth())
            fills = match_orders(orders, od, position, product)
            for fill in fills:
                cash[product] -= fill.price * fill.quantity
                own_trades_prev[product].append(fill)

        for _, row in tick_df.iterrows():
            product    = row["product"]
            mid        = row["mid_price"]
            pos        = position[product]
            pnl        = cash[product] + pos * mid
            records.append({
                "global_ts": global_ts,
                "timestamp": timestamp,
                "day":       row["day"],
                "product":   product,
                "mid_price": mid,
                "position":  pos,
                "cash":      cash[product],
                "pnl":       pnl,
            })

    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

DAY_BOUNDARY = 1_000_000
TITLE_KW  = dict(color="white", fontsize=11, fontweight="bold", pad=6)
LABEL_KW  = dict(color="#aaaaaa", fontsize=9)
TICK_KW   = dict(colors="#888888", labelsize=8)
GRID_KW   = dict(color="#2a2a3a", linewidth=0.5)
SPINE_CLR = "#2a2a3a"


def fmt_ts(x, _):
    if x < DAY_BOUNDARY:
        return f"D-2 {int(x/1000)}k" if x >= 1000 else f"D-2 {int(x)}"
    ts = x - DAY_BOUNDARY
    return f"D-1 {int(ts/1000)}k" if ts >= 1000 else f"D-1 {int(ts)}"


def style_ax(ax, title=""):
    ax.set_facecolor("#1a1b26")
    ax.set_title(title, **TITLE_KW)
    ax.tick_params(axis="both", **TICK_KW)
    ax.xaxis.label.set(**LABEL_KW)
    ax.yaxis.label.set(**LABEL_KW)
    ax.grid(True, **GRID_KW)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)


def apply_ts_fmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_ts))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200_000))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


def add_day_divider(ax):
    ymin, ymax = ax.get_ylim()
    ax.axvline(DAY_BOUNDARY, color="#555577", linewidth=1.0, linestyle="--")
    for mid, lbl in [(DAY_BOUNDARY * 0.5, "Day −2"), (DAY_BOUNDARY * 1.5, "Day −1")]:
        ax.text(mid, ymin + (ymax - ymin) * 0.02, lbl,
                color="#777799", fontsize=7, ha="center", va="bottom")


def plot_results(results: pd.DataFrame, prices_df: pd.DataFrame):
    """
    Per product (one row each):
      Left  – mid-price with buy (▲) / sell (▼) fill markers
      Right – position over time (with ±limit lines)
    Bottom row (full width): cumulative PnL per product + combined
    """
    n = len(PRODUCTS)
    fig = plt.figure(figsize=(16, 5 * (n + 1)))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(n + 1, 2, figure=fig,
                           hspace=0.55, wspace=0.30,
                           top=0.94, bottom=0.05,
                           left=0.07, right=0.97)

    for row_idx, product in enumerate(PRODUCTS):
        color = COLORS[product]
        pr  = prices_df[prices_df["product"] == product].sort_values("global_ts")
        res = results[results["product"] == product].sort_values("global_ts")

        # Position diff → fill markers
        res = res.copy()
        res["pos_diff"] = res["position"].diff().fillna(res["position"].iloc[0])
        buys  = res[res["pos_diff"] > 0]
        sells = res[res["pos_diff"] < 0]

        # Left: mid-price + fills
        ax_p = fig.add_subplot(gs[row_idx, 0])
        ax_p.plot(pr["global_ts"], pr["mid_price"],
                  color=color, linewidth=1.0, alpha=0.55, label="Mid price")
        ax_p.scatter(buys["global_ts"],  buys["mid_price"],
                     marker="^", color="#00ff99", s=45, zorder=5,
                     linewidths=0, label="Buy fill")
        ax_p.scatter(sells["global_ts"], sells["mid_price"],
                     marker="v", color="#ff4466", s=45, zorder=5,
                     linewidths=0, label="Sell fill")
        style_ax(ax_p, f"{product} – Mid Price + Fills")
        ax_p.set_ylabel("Price")
        ax_p.legend(fontsize=7, facecolor="#1a1b26",
                    labelcolor="white", edgecolor=SPINE_CLR)
        apply_ts_fmt(ax_p)
        add_day_divider(ax_p)

        # Right: position
        ax_pos = fig.add_subplot(gs[row_idx, 1])
        ax_pos.plot(res["global_ts"], res["position"],
                    color=color, linewidth=1.2)
        ax_pos.axhline(0,               color="#555577", linewidth=0.8, linestyle="--")
        ax_pos.axhline( POSITION_LIMIT, color="#ff4466", linewidth=0.8,
                        linestyle=":", alpha=0.8, label=f"+{POSITION_LIMIT} limit")
        ax_pos.axhline(-POSITION_LIMIT, color="#ff4466", linewidth=0.8,
                        linestyle=":", alpha=0.8, label=f"-{POSITION_LIMIT} limit")
        style_ax(ax_pos, f"{product} – Position")
        ax_pos.set_ylabel("Units held")
        ax_pos.legend(fontsize=7, facecolor="#1a1b26",
                      labelcolor="white", edgecolor=SPINE_CLR)
        apply_ts_fmt(ax_pos)
        add_day_divider(ax_pos)

    # Bottom: PnL
    ax_pnl = fig.add_subplot(gs[n, :])
    combined = None
    for product in PRODUCTS:
        res = results[results["product"] == product].sort_values("global_ts")
        ax_pnl.plot(res["global_ts"], res["pnl"],
                    color=COLORS[product], linewidth=1.2, label=product)
        s = res.set_index("global_ts")["pnl"]
        combined = s if combined is None else combined.add(s, fill_value=0)

    ax_pnl.plot(combined.index, combined.values,
                color="white", linewidth=1.8, linestyle="--", label="Combined")
    ax_pnl.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
    style_ax(ax_pnl, "Cumulative PnL")
    ax_pnl.set_ylabel("PnL (seashells)")
    ax_pnl.legend(fontsize=8, facecolor="#1a1b26",
                  labelcolor="white", edgecolor=SPINE_CLR)
    apply_ts_fmt(ax_pnl)
    add_day_divider(ax_pnl)

    # Title with final PnL summary
    final = results.groupby("product")["pnl"].last()
    total = final.sum()
    parts = "  |  ".join(f"{p}: {final.get(p, 0):+.0f}" for p in PRODUCTS)
    fig.suptitle(f"Backtest  ·  {parts}  |  Total: {total:+.0f}",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_FILE.resolve()}")
    plt.show()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(DATA_DIR))

    trader_mod = importlib.import_module(TRADER_MODULE)
    trader     = trader_mod.Trader()

    print("Running backtest...")
    prices_df = load_prices()
    results   = run_backtest(trader)

    print("\n── Final PnL ──────────────────────────────────────")
    for product in PRODUCTS:
        pnl = results[results["product"] == product]["pnl"].iloc[-1]
        print(f"  {product:12s}: {pnl:+.2f}")
    total = results.groupby("product")["pnl"].last().sum()
    print(f"  {'TOTAL':12s}: {total:+.2f}")
    print()

    plot_results(results, prices_df)