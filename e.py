import json
import numpy as np
import math
from datamodel import OrderDepth, TradingState, Order, Symbol
from typing import Any

class Trader:

    def __init__(self):
        self.price_history = {
            "TOMATOES": [],
            "EMERALDS": [],
        }
        self.position_limit = {
            "TOMATOES": 80,
            "EMERALDS": 80,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        return (best_bid + best_ask) / 2

    def get_maxamt_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        max_bid_prc = max(order_depth.buy_orders, key=lambda p: order_depth.buy_orders[p])
        max_ask_prc = min(order_depth.sell_orders, key=lambda p: abs(order_depth.sell_orders[p]))
        return (max_bid_prc + max_ask_prc) / 2

    def arb(self, product, order_depth, fair_price, position):
        orders = []
        limit = self.position_limit[product]

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

    def mm(self, product, order_depth, fair_price, position, gamma=0.1, order_amount=20):
        limit = self.position_limit[product]
        orders = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)

        q = position / order_amount
        sigma = 0.5

        kappa_b = 1 / max((fair_price - best_bid) - 1, 1)
        kappa_a = 1 / max((best_ask - fair_price) - 1, 1)

        delta_b = (1/gamma * math.log(1 + gamma/kappa_b)
                   + (2*q + 1)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_b*0.25)
                   * (1 + gamma/kappa_b)**(1 + kappa_b/gamma)))
        delta_a = (1/gamma * math.log(1 + gamma/kappa_a)
                   + (1 - 2*q)/2 * math.sqrt(sigma**2 * gamma / (2*kappa_a*0.25)
                   * (1 + gamma/kappa_a)**(1 + kappa_a/gamma)))

        p_b = min(round(fair_price - delta_b), fair_price, best_bid + 1)
        p_a = max(round(fair_price + delta_a), fair_price, best_ask - 1)

        buy_amt  = min(order_amount, limit - position)
        sell_amt = min(order_amount, limit + position)

        if buy_amt > 0:
            orders.append(Order(product, int(p_b), buy_amt))
        if sell_amt > 0:
            orders.append(Order(product, int(p_a), -sell_amt))

        return orders

    def ema_fair_price(self, product, raw_mid, span=50):
        hist = self.price_history[product]
        if len(hist) < 10:
            return raw_mid
        series = np.array(hist[-200:], dtype=float)
        alpha = 2 / (span + 1)
        ema = series[0]
        for p in series[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return 0.45 * raw_mid + 0.55 * ema

    # ── per-product strategies ────────────────────────────────────────────────

    def trade_emeralds(self, order_depth, position):
        fv = 10000
        limit = self.position_limit["EMERALDS"]
        orders = []

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        buy_qty  = limit - position
        sell_qty = limit + position

        # arb: take any ask at or below fair value
        if best_ask and best_ask <= fv:
            qty = min(-order_depth.sell_orders[best_ask], buy_qty)
            if qty > 0:
                orders.append(Order("EMERALDS", best_ask, qty))
                buy_qty -= qty

        # arb: hit any bid at or above fair value
        if best_bid and best_bid >= fv:
            qty = min(order_depth.buy_orders[best_bid], sell_qty)
            if qty > 0:
                orders.append(Order("EMERALDS", best_bid, -qty))
                sell_qty -= qty

        # passive quotes at 9993 / 10007
        if buy_qty > 0:
            orders.append(Order("EMERALDS", fv - 7, buy_qty))
        if sell_qty > 0:
            orders.append(Order("EMERALDS", fv + 7, -sell_qty))

        return orders

    def trade_tomatoes(self, order_depth, position):
        raw_mid = self.get_maxamt_mid(order_depth)
        if raw_mid is None:
            return []
        fair_price = round(self.ema_fair_price("TOMATOES", raw_mid, span=69))
        orders, position = self.arb("TOMATOES", order_depth, fair_price, position)
        orders += self.mm("TOMATOES", order_depth, fair_price, position, gamma=0.02, order_amount=20)
        return orders

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self.get_mid(order_depth)
            if mid is not None:
                self.price_history[product].append(mid)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(order_depth, position)

        traderData = ""
        conversions = 0
        return result, conversions, traderData