from datamodel import OrderDepth, TradingState, Order
from typing import List
import json
import numpy as np


LIMIT = 80
N = 3
CAP = 20

class Trader:

    def bid(self):
        return 15

    def ema_fair_price(self, product, raw_mid, price_history, span=50):
        hist = price_history[product]
        if len(hist) < 10:
            return raw_mid
        series = np.array(hist[-200:], dtype=float)
        alpha = 2 / (span + 1)
        ema = series[0]
        for p in series[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return 0.7 * raw_mid + 0.3 * ema

    def get_maxamt_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        max_bid_prc = max(order_depth.buy_orders, key=lambda p: order_depth.buy_orders[p])
        max_ask_prc = min(order_depth.sell_orders, key=lambda p: abs(order_depth.sell_orders[p]))
        return (max_bid_prc + max_ask_prc) / 2

    def trade_tomatoes(self, order_depth, position, price_history):
        fv = round(self.ema_fair_price("TOMATOES",
            self.get_maxamt_mid(order_depth) or 5000,
            price_history))
        orders = []

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        buy_qty  = LIMIT - position
        sell_qty = LIMIT + position

        # arb
        if best_ask and best_ask < fv:
            qty = min(-order_depth.sell_orders[best_ask], buy_qty)
            if qty > 0:
                orders.append(Order("TOMATOES", best_ask, qty))
                buy_qty -= qty

        if best_bid and best_bid > fv:
            qty = min(order_depth.buy_orders[best_bid], sell_qty)
            if qty > 0:
                orders.append(Order("TOMATOES", best_bid, -qty))
                sell_qty -= qty

        # 13-wide spread
        if buy_qty > 0:
            orders.append(Order("TOMATOES", fv - 6, buy_qty))
        if sell_qty > 0:
            orders.append(Order("TOMATOES", fv + 6, -sell_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        price_history: Dict[str, List[float]] = json.loads(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            buy_qty  = LIMIT - pos
            sell_qty = LIMIT + pos


            # Update rolling N-period mid price history
            if best_bid is not None and best_ask is not None:
                history = price_history.get(product, [])
                history.append((best_bid + best_ask) / 2)
                price_history[product] = history[-N:]


            if product == "EMERALDS":
                fv  = 10000

                # buy at 10k when possible
                if best_ask and best_ask <= fv:
                    qty = min(-order_depth.sell_orders[best_ask], buy_qty)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_qty -= qty

                if best_bid and best_bid >= fv:
                    qty = min(order_depth.buy_orders[best_bid], sell_qty)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_qty -= qty

                # spread 9993 - 10007
                if buy_qty > 0:
                    orders.append(Order(product, fv - 7, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, fv + 7, -sell_qty))

                result[product] = orders

            if product == "TOMATOES":
                # ====== PRETTY SHIT ======
                # if len(history) < 2:
                #     continue

                # fv = sum(history) / len(history)

                # # buy dips, sell spikes
                # if best_ask is not None and best_ask < fv and buy_qty > 0 and pos < CAP:
                #     qty = min(-order_depth.sell_orders[best_ask], buy_qty)
                #     if qty > 0:
                #         orders.append(Order(product, best_ask, qty))
                #         buy_qty -= qty

                # if best_bid is not None and best_bid > fv and sell_qty > 0 and pos > -CAP:
                #     qty = min(order_depth.buy_orders[best_bid], sell_qty)
                #     if qty > 0:
                #         orders.append(Order(product, best_bid, -qty))
                #         sell_qty -= qty

                # # passive spread
                # if buy_qty > 0:
                #     orders.append(Order(product, int(fv) - 6, buy_qty))
                # if sell_qty > 0:
                #     orders.append(Order(product, int(fv) + 6, -sell_qty))

                # ===== DOESNT WORK ====

                # if len(history) < N:
                #     result[product] = []
                #     continue

                # fv = sum(history) / len(history)

                # # Drift detection: compare short vs. long EMA
                # ema_fast = sum(history[-3:]) / 3        # 3-tick EMA
                # ema_slow = sum(history) / len(history)  # N-tick EMA
                # drift_signal = ema_fast - ema_slow      # positive = uptrend, negative = downtrend

                # # Asymmetric quoting: tighten the quote on the side that reduces inventory
                # # If drifting DOWN, be less eager to buy (raise our buy threshold)
                # QUOTE_HALF = 5   # quote 5 pts either side of FV (inside the 13-pt market spread)
                # DRIFT_LEAN = min(abs(drift_signal), 3)  # cap the lean

                # if drift_signal < 0:  # downtrend — lean short
                #     my_bid = fv - QUOTE_HALF - DRIFT_LEAN   # less aggressive buy
                #     my_ask = fv + QUOTE_HALF                # normal sell
                # else:  # uptrend — lean long
                #     my_bid = fv - QUOTE_HALF
                #     my_ask = fv + QUOTE_HALF + DRIFT_LEAN

                # # hard skew as limits approach
                # if pos > 10:   # getting long, stop buying, push ask down
                #     my_ask = fv + 2
                #     my_bid = fv - 10   # stop buying
                # elif pos < -10:  # getting short, stop selling
                #     my_bid = fv - 2
                #     my_ask = fv + 10

                # # Place orders
                # if best_ask <= my_bid and buy_qty > 0:
                #     orders.append(Order(product, best_ask, min(buy_qty, 5)))
                # if best_bid >= my_ask and sell_qty > 0:
                #     orders.append(Order(product, best_bid, -min(sell_qty, 5)))

                # print(f"{product} | fv={fv:.1f} my_bid={my_bid:.1f} my_ask={my_ask:.1f} | best_bid={best_bid} best_ask={best_ask} | pos={pos}")

                # ====== COOKED =======

                # total_bid_vol = sum(order_depth.buy_orders.values())
                # total_ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())

                # imbalance_key = product + "_imbalance"
                # state_data = price_history.get(imbalance_key, {"cum_bid": 0, "cum_ask": 0})
                # state_data["cum_bid"] += total_bid_vol
                # state_data["cum_ask"] += total_ask_vol
                # price_history[imbalance_key] = state_data

                # imbalance = state_data["cum_bid"] - state_data["cum_ask"]

                # if imbalance > 0 and sell_qty > 0:
                #     # More bid volume -> price likely to fall
                #     orders.append(Order(product, best_bid, -sell_qty))
                # elif imbalance < 0 and buy_qty > 0:
                #     # More ask volume -> price likely to rise
                #     orders.append(Order(product, best_ask, buy_qty))

                # print(f"{product} | imbalance={imbalance} pos={pos} bid_vol={total_bid_vol} ask_vol={total_ask_vol}")

                # ===== Aight on back tester... doesnt do shit on the actual thing (17) aghhhhhhh =====
                # mid = (best_bid + best_ask) / 2
                # history = price_history.get('TOMATOES_hist', [])
                # history.append(mid)
                # price_history['TOMATOES_hist'] = history[-2*N:]

                # # Drift detection
                # if len(history) >= N:
                #     drift = history[-1] - history[-N]
                # else:
                #     drift = 0

                # # Skew quotes based on drift and current position
                # # Trending down: widen buy side, tighten sell side
                # # Trending up: widen sell side, tighten buy side
                # skew = int(np.clip(drift * 0.5, -3, 3))  # gentle skew

                # buy_price  = best_bid - max(0,  skew)
                # sell_price = best_ask + max(0, -skew)

                # if buy_qty > 0:
                #     orders.append(Order(product, buy_price, buy_qty))
                # if sell_qty > 0:
                #     orders.append(Order(product, sell_price, -sell_qty))
                result[product] = self.trade_tomatoes(order_depth, pos, price_history)



        return result, 0, json.dumps(price_history)

