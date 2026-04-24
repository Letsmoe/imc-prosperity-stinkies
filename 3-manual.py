import numpy as np

def payoff(p, mu, N=1000):
    capture = max(0, np.ceil((p - 795) / 5))
    profit = 920 - p
    if profit <= 0:
        return 0
    penalty = min(((920 - mu) / (920 - p))**3, 1)
    return N * capture * profit * penalty

# 1. Sweep p assuming Nash (mu = p, penalty = 1)
print("=== Nash Sweep (mu = p) ===")
best_p, best_f = 0, 0
for p in range(796, 920, 1):
    f = payoff(p, p)
    if f > best_f:
        best_f = f
        best_p = p

print(f"Nash optimal bid: {best_p}, payoff: {best_f}")

# 2. Check around the peak
print("\n=== Around the peak ===")
for p in range(671, 920)[::5]:
    print(f"p={p}, payoff={payoff(p, p):.1f}")

# 3. What if you deviate and others bid 856?
print("\n=== Deviation from Nash (others bid 856) ===")
mu = 856
for p in range(671, 920)[::5]:
    print(f"p={p}, mu={mu}, payoff={payoff(p, mu):.1f}")