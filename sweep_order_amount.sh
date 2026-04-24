#!/bin/bash

TRADER_FILE="e.py"
ROUND=1
RESULTS_FILE="sweep_results.txt"

> "$RESULTS_FILE"

# for amount in $(seq 0 5 80); do
#     # Patch order_amount in the trader file
#     sed "184s/order_amount=[0-9]*/order_amount=$amount/" "$TRADER_FILE" > "tmp_trader.py"

#     # Run backtest and capture output
#     output=$(prosperity4btest tmp_trader.py $ROUND 2>&1)

#     # Extract final PnL — adjust grep pattern if your backtester outputs differently
#     pnl=$(echo "$output" | grep "Total profit:" | tail -1 | grep -oE '[0-9,]+' | tr -d ',')

#     echo "order_amount=$amount -> PnL=$pnl"
#     echo "order_amount=$amount -> PnL=$pnl" >> "$RESULTS_FILE"
# done


for amount in $(seq 15 1 25); do
    # Patch order_amount in the trader file
    sed "184s/order_amount=[0-9]*/order_amount=$amount/" "$TRADER_FILE" > "tmp_trader.py"

    # Run backtest and capture output
    output=$(prosperity4btest tmp_trader.py $ROUND 2>&1)

    # Extract final PnL — adjust grep pattern if your backtester outputs differently
    pnl=$(echo "$output" | grep "Total profit:" | tail -1 | grep -oE '[0-9,]+' | tr -d ',')

    echo "order_amount=$amount -> PnL=$pnl"
    echo "order_amount=$amount -> PnL=$pnl" >> "$RESULTS_FILE"
done

rm -f tmp_trader.py

echo ""
echo "Best result:"
sort -t= -k3 -n "$RESULTS_FILE" | tail -1