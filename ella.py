from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:

    def bid(self):
        return 15
    
    # def run(self, state: TradingState):
    #     """Only method required. It takes all buy and sell orders for all
    #     symbols as an input, and outputs a list of orders to be sent."""

    #     print("traderData: " + state.traderData)
    #     print("Observations: " + str(state.observations))

    #     # Orders to be placed on exchange matching engine
    #     result = {}
    #     for product in state.order_depths:
    #         order_depth: OrderDepth = state.order_depths[product]
    #         orders: List[Order] = []

    #         if product == "EMERALDS":
    #             acceptable_price = 10000
    #         else:
    #             continue
    #         print("Acceptable price : " + str(acceptable_price))
    #         print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

    #         if len(order_depth.sell_orders) != 0:
    #             best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
    #             if int(best_ask) <= acceptable_price:
    #                 print("BUY", str(-best_ask_amount) + "x", best_ask)
    #                 orders.append(Order(product, best_ask, -best_ask_amount))

    #         if len(order_depth.buy_orders) != 0:
    #             best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
    #             if int(best_bid) > acceptable_price:
    #                 print("SELL", str(best_bid_amount) + "x", best_bid)
    #                 orders.append(Order(product, best_bid-1, -best_bid_amount))

    #         result[product] = orders

    #     traderData = ""  # No state needed - we check position directly
    #     conversions = 0
    #     return result, conversions, traderData

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "EMERALDS":
                fair  = 10000
                limit = 80

                # buy at 10k when possible
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask <= fair:
                        qty = min(-order_depth.sell_orders[best_ask], limit - pos)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))

                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid >= fair:
                        qty = min(order_depth.buy_orders[best_bid], limit + pos)
                        if qty > 0:
                            orders.append(Order(product, best_bid, -qty))

                # spread at 9993/10007
                buy_qty  = limit - pos
                sell_qty = limit + pos

                if buy_qty > 0:
                    orders.append(Order(product, fair - 7, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, fair + 7, -sell_qty))

            result[product] = orders

        return result, 0, ""
if __name__ == "__main__":
    trader = Trader()
    result, conversions, traderData = trader.run()