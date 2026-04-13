from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT = 80
N = 7

class Trader:

    def bid(self):
        return 15


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


            # if product == "EMERALDS":
            #     fair  = 10000

            #     # buy at 10k when possible
            #     if best_ask and best_ask <= fair:
            #         qty = min(-order_depth.sell_orders[best_ask], buy_qty)
            #         if qty > 0:
            #             orders.append(Order(product, best_ask, qty))
            #             buy_qty -= qty

            #     if best_bid and best_bid >= fair:
            #         qty = min(order_depth.buy_orders[best_bid], sell_qty)
            #         if qty > 0:
            #             orders.append(Order(product, best_bid, -qty))
            #             sell_qty -= qty

            #     # spread 9993 - 10007
            #     if buy_qty > 0:
            #         orders.append(Order(product, fair - 7, buy_qty))
            #     if sell_qty > 0:
            #         orders.append(Order(product, fair + 7, -sell_qty))

            if product == "TOMATOES":
                if len(history) < 2:
                    continue

                fair = sum(history) / len(history)

                # buy dips, sell spikes
                if best_ask is not None and best_ask < fair and buy_qty > 0 and pos < 20:
                    qty = min(-order_depth.sell_orders[best_ask], buy_qty)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_qty -= qty

                if best_bid is not None and best_bid > fair and sell_qty > 0 and pos > -20:
                    qty = min(order_depth.buy_orders[best_bid], sell_qty)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_qty -= qty

                # passive spread
                if buy_qty > 0:
                    orders.append(Order(product, int(fair) - 6, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, int(fair) + 6, -sell_qty))

            result[product] = orders

        return result, 0, json.dumps(price_history)

