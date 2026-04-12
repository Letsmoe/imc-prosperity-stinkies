from trader import Trader
import json
import csv

with open('./data/prices_round_0_day_-1.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    data = list(csv.DictReader(csvfile, delimiter=";"))
    
print(data)


moritz_trader = Trader()