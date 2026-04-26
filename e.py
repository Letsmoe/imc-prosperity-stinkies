import json
import numpy as np
import math
# from scipy.stats import norm
# from scipy.optimize import brentq
from datamodel import OrderDepth, TradingState, Order
from typing import Any

# ── Black-Scholes helpers ─────────────────────────────────────────────────────

# def bs_call(S, K, T, sigma, r=0.0):
#     if sigma <= 0 or T <= 0:
#         return max(S - K, 0.0)
#     d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
#     d2 = d1 - sigma * math.sqrt(T)
#     return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# def bs_delta(S, K, T, sigma, r=0.0):
#     if sigma <= 0 or T <= 0:
#         return 1.0 if S > K else 0.0
#     d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
#     return norm.cdf(d1)

# def implied_vol(S, K, T, market_price, r=0.0):
#     intrinsic = max(S - K, 0.0)
#     if market_price <= intrinsic + 1e-6:
#         return None
#     try:
#         return brentq(lambda sig: bs_call(S, K, T, sig, r) - market_price, 1e-6, 10.0)
#     except Exception:
#         return None


class Trader:

    # Round 3: TTE = 5 days at day 0 tick 0
    ROUND_START_TTE = 5.0

    VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

    # IV smile seeds from data (T-corrected)
    IV_SEED = {
        4000: 0.0609,
        4500: 0.0341,
        5000: 0.0175,
        5100: 0.0173,
        5200: 0.0175,
        5300: 0.0177,
        5400: 0.0166,
        5500: 0.0180,
        6000: 0.0286,
        6500: 0.0434,
    }

    POSITION_LIMIT = {
        "TOMATOES": 80,
        "EMERALDS": 80,
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
        "VELVETFRUIT_EXTRACT": 200,
        "HYDROGEL_PACK": 200,
        **{f"VEV_{K}": 300 for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
    }

    # ── state serialisation ───────────────────────────────────────────────────

    def load_state(self, traderData: str):
        """Restore persisted state from traderData string."""
        default = {
            "price_history": {p: [] for p in list(self.POSITION_LIMIT.keys())},
            "iv_ema": {str(K): v for K, v in self.IV_SEED.items()},
            "day": 0,
            "last_ts": -1,
        }
        if not traderData:
            return default
        try:
            return json.loads(traderData)
        except Exception:
            return default

    def save_state(self, state: dict) -> str:
        # Keep price history bounded to last 500 ticks per product
        for k in state["price_history"]:
            state["price_history"][k] = state["price_history"][k][-500:]
        return json.dumps(state)

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2

    def get_maxamt_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        max_bid = max(order_depth.buy_orders, key=lambda p: order_depth.buy_orders[p])
        max_ask = max(order_depth.sell_orders, key=lambda p: abs(order_depth.sell_orders[p]))  # max, not min
        return (max_bid + max_ask) / 2

    def get_vwap_mid(self, order_depth: OrderDepth) -> float | None:
        """
        VWAP mid: volume-weighted average price across all book levels,
        computed separately for bid and ask sides, then averaged.
        Outperforms simple get_mid as a fair value estimator (~3% lower MAE
        vs future mid).
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        bid_notional = sum(p * v for p, v in order_depth.buy_orders.items())
        bid_volume   = sum(v     for v in order_depth.buy_orders.values())

        ask_notional = sum(p * abs(v) for p, v in order_depth.sell_orders.items())
        ask_volume   = sum(abs(v)     for v in order_depth.sell_orders.values())

        if bid_volume == 0 or ask_volume == 0:
            return None

        vwap_bid = bid_notional / bid_volume
        vwap_ask = ask_notional / ask_volume

        return (vwap_bid + vwap_ask) / 2

    def ema(self, series, span):
        alpha = 2 / (span + 1)
        val = series[0]
        for p in series[1:]:
            val = alpha * p + (1 - alpha) * val
        return val

    def ema_fair_price(self, hist, raw_mid, span=50, n=200, s=0.5):
        if len(hist) < span:
            return raw_mid
        series = hist[-n:]
        return round(s * raw_mid + (1 - s) * self.ema(series, span))

    def get_tte(self, day: int, timestamp: int) -> float:
        """TTE in days. Round 3 day 0 tick 0 = 5.0."""
        return max((self.ROUND_START_TTE - day) - timestamp / 1_000_000, 1e-4)

    def arb(self, product, order_depth, fair_price, position):
        orders = []
        limit = self.POSITION_LIMIT[product]

        for ask_price in sorted(order_depth.sell_orders):
            ask_amt = -order_depth.sell_orders[ask_price]
            if ask_price < fair_price:
                buy_amt = min(ask_amt, limit - position)
                if buy_amt > 0:
                    orders.append(Order(product, ask_price, buy_amt))
                    position += buy_amt
            elif ask_price == fair_price and position < 0:
                buy_amt = min(ask_amt, -position)
                orders.append(Order(product, ask_price, buy_amt))
                position += buy_amt

        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            bid_amt = order_depth.buy_orders[bid_price]
            if bid_price > fair_price:
                sell_amt = min(bid_amt, limit + position)
                if sell_amt > 0:
                    orders.append(Order(product, bid_price, -sell_amt))
                    position -= sell_amt
            elif bid_price == fair_price and position > 0:
                sell_amt = min(bid_amt, position)
                orders.append(Order(product, bid_price, -sell_amt))
                position -= sell_amt

        return orders, position

    def mm(self, product, order_depth, fair_price, position, gamma=0.1, order_amount=20, adj=1, side=None):
        limit = self.POSITION_LIMIT[product]
        orders = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)

        q = position / order_amount
        sigma = 0.5

        # kappa_b = 1 / max((fair_price - best_bid) - 1, 1)
        # kappa_a = 1 / max((best_ask - fair_price) - 1, 1)

        # delta_b = (1/gamma * math.log(1 + gamma/kappa_b)
        #            + (2*q + 1)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_b*0.25)
        #            * (1 + gamma/kappa_b)**(1 + kappa_b/gamma)))
        # delta_a = (1/gamma * math.log(1 + gamma/kappa_a)
        #            + (1 - 2*q)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_a*0.25)
        #            * (1 + gamma/kappa_a)**(1 + kappa_a/gamma)))

        # p_b = min(round(fair_price - delta_b), fair_price, best_bid + adj)
        # p_a = max(round(fair_price + delta_a), fair_price, best_ask - adj)

        # buy_amt  = min(order_amount, limit - position)
        # sell_amt = min(order_amount, limit + position)

        # if buy_amt > 0:
        #     orders.append(Order(product, int(p_b), buy_amt))
        # if sell_amt > 0:
        #     orders.append(Order(product, int(p_a), -sell_amt))

        if side is None or side == "bid":
            kappa_b = 1 / max((fair_price - best_bid) - 1, 1)
            delta_b = (1/gamma * math.log(1 + gamma/kappa_b)
                       + (2*q + 1)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_b*0.25)
                       * (1 + gamma/kappa_b)**(1 + kappa_b/gamma)))
            # p_b = min(round(fair_price - delta_b), fair_price, best_bid + adj)
            p_b = min(round(fair_price - delta_b), fair_price, best_bid - 1)  # outside bot A
            buy_amt = min(order_amount, limit - position)
            if buy_amt > 0:
                orders.append(Order(product, int(p_b), buy_amt))

        if side is None or side == "ask":
            kappa_a = 1 / max((best_ask - fair_price) - 1, 1)
            delta_a = (1/gamma * math.log(1 + gamma/kappa_a)
                       + (1 - 2*q)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_a*0.25)
                       * (1 + gamma/kappa_a)**(1 + kappa_a/gamma)))
            # p_a = max(round(fair_price + delta_a), fair_price, best_ask - adj)
            p_a = max(round(fair_price + delta_a), fair_price, best_ask + 1)  # outside bot A
            sell_amt = min(order_amount, limit + position)
            if sell_amt > 0:
                orders.append(Order(product, int(p_a), -sell_amt))


        return orders

    # ── per-product strategies ────────────────────────────────────────────────

    def trade_emeralds(self, order_depth, position):
        fv = 10000
        limit = self.POSITION_LIMIT["EMERALDS"]
        orders = []

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        buy_qty  = limit - position
        sell_qty = limit + position

        if best_ask and best_ask <= fv:
            qty = min(-order_depth.sell_orders[best_ask], buy_qty)
            if qty > 0:
                orders.append(Order("EMERALDS", best_ask, qty))
                buy_qty -= qty

        if best_bid and best_bid >= fv:
            qty = min(order_depth.buy_orders[best_bid], sell_qty)
            if qty > 0:
                orders.append(Order("EMERALDS", best_bid, -qty))
                sell_qty -= qty

        if buy_qty > 0:
            orders.append(Order("EMERALDS", fv - 7, buy_qty))
        if sell_qty > 0:
            orders.append(Order("EMERALDS", fv + 7, -sell_qty))

        return orders

    def trade_tomatoes(self, order_depth, position, hist):
        raw_mid = self.get_maxamt_mid(order_depth)
        if raw_mid is None:
            return []
        fair_price = round(self.ema_fair_price(hist, raw_mid, span=69))
        orders, position = self.arb("TOMATOES", order_depth, fair_price, position)
        orders += self.mm("TOMATOES", order_depth, fair_price, position, gamma=0.02, order_amount=20)
        return orders

    def trade_coated_osmium(self, order_depth, position, hist):
        fair = self.ema_fair_price(hist, 10001, span=3, n=13, s=0.53)
        orders, position = self.arb("ASH_COATED_OSMIUM", order_depth, fair, position)
        orders += self.mm("ASH_COATED_OSMIUM", order_depth, fair, position,
                          gamma=0.05, order_amount=20)
        return orders

    def trade_pepper(self, order_depth, position, hist, timestamp):
        orders = []
        limit = self.POSITION_LIMIT["INTARIAN_PEPPER_ROOT"]
        if position >= limit:
            return orders
        remaining = limit - position
        n = len(hist)
        if n < 20:
            fv = round(hist[-1]) if hist else None
        else:
            current_tick = timestamp // 100
            ticks = list(range(current_tick - n + 1, current_tick + 1))
            prices = hist
            slope, intercept = np.polyfit(ticks, prices, 1)
            fv = round(intercept + slope * current_tick)
        if fv is None:
            return orders
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if remaining <= 0 or ask_price >= fv + 9:
                break
            buy_amt = min(-order_depth.sell_orders[ask_price], remaining)
            if buy_amt > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, buy_amt))
                remaining -= buy_amt
        return orders

    def trade_velvetfruit(self, order_depth, position, hist):
        # mid = self.get_mid(order_depth)
        mid = self.get_vwap_mid(order_depth)
        if mid is None:
            return []
        fair = self.ema_fair_price(hist, mid, span=50, n=300, s=0.4)
        orders, pos = self.arb("VELVETFRUIT_EXTRACT", order_depth, fair, position)
        orders += self.mm("VELVETFRUIT_EXTRACT", order_depth, fair, pos,
                          gamma=0.03, order_amount=35, adj=1)
        return orders

    # def trade_hydrogel(self, order_depth, position, hist):
    #     # mid = self.get_vwap_mid(order_depth)
    #     mid = self.get_mid(order_depth)
    #     if mid is None:
    #         return []
    #     fair = self.ema_fair_price(hist, mid, span=20, n=100, s=0.39)
    #     orders, pos = self.arb("HYDROGEL_PACK", order_depth, fair, position)
    #     orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #                       gamma=0.05, order_amount=151, adj=1)
    #     return orders

    # def trade_hydrogel(self, order_depth, position, hist):
    #     # mid = self.get_mid(order_depth)
    #     mid = self.get_vwap_mid(order_depth)
    #     if mid is None:
    #         return []

    #     fair = self.ema_fair_price(hist, mid, span=20, n=100, s=0.4)

    #     # sensitivity - 1.0 ~ 3.0
    #     # Lower catches trends earlier but will false-positive on normal volatility
    #     # Higher is more conservative
    #     s = 1.8
    #     # lookback window - shorter reacts faster, at the cost of more noise
    #     lw = 10

    #     # trend filter
    #     if len(hist) >= lw:
    #         hist_arr = np.array(hist[-lw:])
    #         trend = mid - hist_arr.mean()
    #         sigma = hist_arr.std()
    #         trending = sigma > 0 and abs(trend) > s * sigma
    #         signed_trend = mid - np.array(hist[-lw:]).mean()
    #     else:
    #         trending = False

    #     # orders, pos = [], position
    #     # if not trending:
    #     #     orders, pos = self.arb("HYDROGEL_PACK", order_depth, fair, position)

    #     # gamma = 0.12 if trending else 0.05
    #     # orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #     #                   gamma=gamma, order_amount=21, adj=1)


    #     # if trending and abs(position) > 100:
    #     #     # fully stop quoting — wait for passive unwind
    #     #     pass
    #     # else:
    #     #     gamma = 0.12 if trending else 0.05
    #     #     orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #     #                       gamma=gamma, order_amount=21, adj=1)

    #     orders, pos = [], position

    #     if not trending:
    #         orders, pos = self.arb("HYDROGEL_PACK", order_depth, fair, position)
    #         orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #                           gamma=0.05, order_amount=21, adj=1)
    #     else:
    #         down_trend = signed_trend < 0  # mid below recent mean

    #         # if down_trend:
    #         #     # falling market: only quote ask (unwind longs, don't add)
    #         #     orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #         #         side="ask", gamma=0.12, order_amount=20, adj=1)
    #         if down_trend and pos > 0:
    #             # hit bids to unwind longs
    #             for bid_p in sorted(order_depth.buy_orders, reverse=True):
    #                 if pos <= 0:
    #                     break
    #                 unwind = min(order_depth.buy_orders[bid_p], pos)
    #                 if unwind > 0:
    #                     orders.append(Order("HYDROGEL_PACK", bid_p, -unwind))
    #                     pos -= unwind
    #         else:
    #             # rising market: only quote bid (unwind shorts, don't add)
    #             orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
    #                 side="bid", gamma=0.12, order_amount=20, adj=1)
    #     return orders

    # ── Intraday drift curve for HYDROGEL_PACK ────────────────────────────────
    # Fitted from 3-day average of prices_round_3 data.
    # Baseline ~9991. Price reliably dips −20 to −30 pts in ts 0–550k,
    # then ramps +20 to +45 pts in ts 600k–950k, settling back near fair by EOD.
    # Drift values (pts above baseline) sampled every 50k ticks — used to build
    # the intraday fair-value adjustment at runtime via linear interpolation.
    _HG_DRIFT_TS    = [     0,  50000, 100000, 150000, 200000, 250000,
                       300000, 350000, 400000, 450000, 500000, 550000,
                       600000, 650000, 700000, 750000, 800000, 850000,
                       900000, 950000, 999900]
    _HG_DRIFT_VAL   = [  -7.7,  -27.0,  -21.6,   -2.8,  -13.6,  -28.9,
                        -17.9,   -4.8,  -29.3,  -25.0,   -5.5,   -2.2,
                         21.7,   17.3,   43.5,   38.1,   16.9,   20.4,
                         11.4,   17.2,    9.0]

    def _hg_drift(self, timestamp: int) -> float:
        """Interpolated intraday drift (pts) for HYDROGEL_PACK at this timestamp."""
        ts_arr  = self._HG_DRIFT_TS
        val_arr = self._HG_DRIFT_VAL
        if timestamp <= ts_arr[0]:
            return val_arr[0]
        if timestamp >= ts_arr[-1]:
            return val_arr[-1]
        for i in range(len(ts_arr) - 1):
            if ts_arr[i] <= timestamp < ts_arr[i + 1]:
                frac = (timestamp - ts_arr[i]) / (ts_arr[i + 1] - ts_arr[i])
                return val_arr[i] + frac * (val_arr[i + 1] - val_arr[i])
        return 0.0

    def trade_hydrogel(self, order_depth, position, hist, timestamp):
        """
        Intraday-drift-aware strategy for HYDROGEL_PACK.

        Fair value = EMA-smoothed mid + intraday drift adjustment.

        Phase logic (driven by the fitted drift curve):
          • ts 0–550k  (drift < 0, price below baseline):
              ACCUMULATE — arb aggressively on the ask side; MM skewed long.
              Target: build up to +LIMIT by ts ~450k.
          • ts 550k–750k (drift rising fast, +20 to +44):
              HOLD / passive unwind — stop buying, let position ride the ramp.
              Only MM on ask side to unwind longs at a premium.
          • ts 750k–999k (drift fading back toward 0):
              UNWIND — sell aggressively to close position before EOD.
        """
        BASELINE   = 9991.0
        LIMIT      = self.POSITION_LIMIT["HYDROGEL_PACK"]   # 200
        orders     = []

        mid = self.get_vwap_mid(order_depth)
        if mid is None:
            return []

        drift = self._hg_drift(timestamp)
        # Fair value shifts with the known intraday drift
        drift_fair = BASELINE + drift
        # Blend with a short EMA of recent mids for microstructure stability
        ema_fair = self.ema_fair_price(hist, mid, span=20, n=80, s=0.39)
        fair = round(0.31 * drift_fair + 0.69 * ema_fair)

        pos = position

        if timestamp < 550_000:
            # ── Accumulation phase: price is cheap, buy aggressively ──────
            # Take every ask below fair; post bids to fill limit.
            buy_cap = LIMIT - pos
            for ask_p in sorted(order_depth.sell_orders):
                if buy_cap <= 0:
                    break
                if ask_p <= fair:
                    qty = min(-order_depth.sell_orders[ask_p], buy_cap)
                    if qty > 0:
                        orders.append(Order("HYDROGEL_PACK", ask_p, qty))
                        pos += qty
                        buy_cap -= qty
            # Also place a passive resting bid just inside fair
            if buy_cap > 0 and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders)
                p_bid = min(fair - 1, best_bid + 1)
                orders.append(Order("HYDROGEL_PACK", int(p_bid), min(30, buy_cap)))

        elif timestamp < 750_000:
            # ── Hold / passive unwind phase: ride the ramp up ─────────────
            # Don't add inventory. Sell into strength above fair.
            sell_cap = LIMIT + pos
            for bid_p in sorted(order_depth.buy_orders, reverse=True):
                if sell_cap <= 0:
                    break
                if bid_p >= fair + 5:   # only sell when clearly above fair
                    qty = min(order_depth.buy_orders[bid_p], sell_cap)
                    if qty > 0:
                        orders.append(Order("HYDROGEL_PACK", bid_p, -qty))
                        pos -= qty
                        sell_cap -= qty
            # Passive ask just above fair to capture more premium
            if sell_cap > 0 and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders)
                p_ask = max(fair + 2, best_ask - 1)
                orders.append(Order("HYDROGEL_PACK", int(p_ask), -min(30, sell_cap)))

        else:
            # ── Unwind phase: drift fading, flatten before EOD ────────────
            if pos > 0:
                # Hit bids to unwind longs; become more aggressive as EOD approaches
                urgency = (timestamp - 750_000) / (999_900 - 750_000)  # 0→1
                min_bid = fair - round(urgency * 15)   # accept up to 15 pts slippage at EOD
                sell_cap = pos
                for bid_p in sorted(order_depth.buy_orders, reverse=True):
                    if sell_cap <= 0:
                        break
                    if bid_p >= min_bid:
                        qty = min(order_depth.buy_orders[bid_p], sell_cap)
                        if qty > 0:
                            orders.append(Order("HYDROGEL_PACK", bid_p, -qty))
                            pos -= qty
                            sell_cap -= qty
                # Passive ask to catch any remaining bots
                if sell_cap > 0 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders)
                    p_ask = max(fair + 1, best_ask - 1)
                    orders.append(Order("HYDROGEL_PACK", int(p_ask), -min(40, sell_cap)))
            elif pos < 0:
                # Shouldn't normally be short, but cover defensively
                buy_cap = -pos
                for ask_p in sorted(order_depth.sell_orders):
                    if buy_cap <= 0:
                        break
                    qty = min(-order_depth.sell_orders[ask_p], buy_cap)
                    if qty > 0:
                        orders.append(Order("HYDROGEL_PACK", ask_p, qty))
                        pos += qty
                        buy_cap -= qty

        return orders


        # mid = self.get_vwap_mid(order_depth)
        # if mid is None:
        #     return []

        # fair = self.ema_fair_price(hist, mid, span=20, n=100, s=0.4)

        # # Deviation from fair in sigma units
        # lw = 20
        # if len(hist) >= lw:
        #     sigma = np.array(hist[-lw:]).std()
        #     dev_sigmas = abs(mid - fair) / sigma if sigma > 0 else 0
        # else:
        #     dev_sigmas = 0

        # # Skew MM harder toward fair when price is far from it.
        # # When mid << fair: AS inventory skew already bids more aggressively (we accumulate
        # # cheap, expect reversion up). When mid >> fair: skews ask side.
        # # The order_amount controls how aggressively q = position/order_amount skews quotes —
        # # shrink it as deviation grows so the skew term amplifies naturally.
        # base_order_amount = 20
        # # At 0 sigma: order_amount=20. At 3+ sigma: order_amount=7 (3x more skew).
        # order_amount = max(7, round(base_order_amount / max(1.0, dev_sigmas)))

        # orders, pos = self.arb("HYDROGEL_PACK", order_depth, fair, position)
        # orders += self.mm("HYDROGEL_PACK", order_depth, fair, pos,
        #                   gamma=0.05, order_amount=order_amount, adj=1)

        # # Emergency unwind only if very offside AND position is on the wrong side of fair.
        # # Flipped from before: long + price ABOVE fair = overextended, trim.
        # #                       short + price BELOW fair = overextended, trim.
        # UNWIND_THRESHOLD = 3.0
        # if dev_sigmas > UNWIND_THRESHOLD:
        #     if position > 0 and mid > fair:
        #         # Long but price above fair — mean reversion expected downward, trim longs
        #         for bid_p in sorted(order_depth.buy_orders, reverse=True):
        #             unwind = min(order_depth.buy_orders[bid_p], pos // 2)
        #             if unwind > 0:
        #                 orders.append(Order("HYDROGEL_PACK", bid_p, -unwind))
        #                 pos -= unwind
        #     elif position < 0 and mid < fair:
        #         # Short but price below fair — mean reversion expected upward, trim shorts
        #         for ask_p in sorted(order_depth.sell_orders):
        #             unwind = min(-order_depth.sell_orders[ask_p], (-pos) // 2)
        #             if unwind > 0:
        #                 orders.append(Order("HYDROGEL_PACK", ask_p, unwind))
        #                 pos += unwind

        # return orders

    # def trade_vev_options(self, state_order_depths, state_positions, spot, iv_ema, day, timestamp):
    #     """IV Scalping across all 10 VEV strikes."""
    #     result = {}
    #     tte = self.get_tte(day, timestamp)

    #     for K in self.VEV_STRIKES:
    #         product = f"VEV_{K}"
    #         order_depth = state_order_depths.get(product)
    #         if order_depth is None:
    #             continue

    #         position = state_positions.get(product, 0)
    #         limit    = self.POSITION_LIMIT[product]
    #         orders   = []

    #         mid = self.get_mid(order_depth)
    #         if mid is None:
    #             continue

    #         iv = implied_vol(spot, K, tte, mid)
    #         if iv is not None:
    #             iv_ema[str(K)] = (0.05 * iv + 0.95 * iv_ema[str(K)])

    #         sigma  = iv_ema[str(K)]
    #         fair   = bs_call(spot, K, tte, sigma)
    #         fair_r = round(fair * 2) / 2

    #         buy_cap  = limit - position
    #         sell_cap = limit + position

    #         for ask_p in sorted(order_depth.sell_orders):
    #             if ask_p >= fair_r or buy_cap <= 0:
    #                 break
    #             qty = min(-order_depth.sell_orders[ask_p], buy_cap)
    #             if qty > 0:
    #                 orders.append(Order(product, ask_p, qty))
    #                 buy_cap  -= qty
    #                 position += qty

    #         for bid_p in sorted(order_depth.buy_orders, reverse=True):
    #             if bid_p <= fair_r or sell_cap <= 0:
    #                 break
    #             qty = min(order_depth.buy_orders[bid_p], sell_cap)
    #             if qty > 0:
    #                 orders.append(Order(product, bid_p, -qty))
    #                 sell_cap -= qty
    #                 position -= qty

    #         best_bid = max(order_depth.buy_orders)  if order_depth.buy_orders  else None
    #         best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

    #         if best_bid is not None and buy_cap > 0:
    #             p_bid = min(int(fair_r) - 1, best_bid + 1)
    #             orders.append(Order(product, p_bid, min(15, buy_cap)))

    #         if best_ask is not None and sell_cap > 0:
    #             p_ask = max(int(math.ceil(fair_r)) + 1, best_ask - 1)
    #             orders.append(Order(product, p_ask, -min(15, sell_cap)))

    #         if orders:
    #             result[product] = orders

    #     return result

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):

        # ── restore persisted state ────────────────────────────────────────
        s = self.load_state(state.traderData)
        hist      = s["price_history"]
        iv_ema    = s["iv_ema"]
        day       = s["day"]
        last_ts   = s["last_ts"]

        # Detect day rollover: timestamp resets to near 0 after each day
        if last_ts > 500000 and state.timestamp < 100000:
            day += 1
        s["last_ts"] = state.timestamp
        s["day"]     = day

        # ── update price histories ─────────────────────────────────────────
        for product, order_depth in state.order_depths.items():
            if product not in hist:
                hist[product] = []
            mid = self.get_mid(order_depth)
            if mid is not None:
                hist[product].append(mid)

        # ── get VEV spot for options pricing ──────────────────────────────
        vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_spot  = self.get_mid(vev_depth) if vev_depth else None
        if vev_spot is None and hist.get("VELVETFRUIT_EXTRACT"):
            vev_spot = hist["VELVETFRUIT_EXTRACT"][-1]

        # ── trade ─────────────────────────────────────────────────────────
        result = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(order_depth, position, hist.get("TOMATOES", []))
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_coated_osmium(order_depth, position, hist.get("ASH_COATED_OSMIUM", []))
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_pepper(order_depth, position, hist.get("INTARIAN_PEPPER_ROOT", []), state.timestamp)
            # elif product == "VELVETFRUIT_EXTRACT":
            #     result[product] = self.trade_velvetfruit(order_depth, position, hist.get("VELVETFRUIT_EXTRACT", []))
            elif product == "HYDROGEL_PACK":
                result[product] = self.trade_hydrogel(order_depth, position, hist.get("HYDROGEL_PACK", []), state.timestamp)

        # if vev_spot is not None:
        #     result.update(self.trade_vev_options(
        #         state.order_depths, state.position, vev_spot, iv_ema, day, state.timestamp
        #     ))

        # ── persist state ─────────────────────────────────────────────────
        s["price_history"] = hist
        s["iv_ema"]        = iv_ema
        traderData = self.save_state(s)

        return result, 0, traderData