# @block setup
from pathlib import Path
import pandas as pd
import pulp

DATA = Path(__file__).parent / "data"

# @block data
products = pd.read_csv(DATA / "products.csv", index_col="product")
periods = pd.read_csv(DATA / "periods.csv", index_col="period")
demand = pd.read_csv(DATA / "demand.csv", index_col=["product", "period"])["demand"].to_dict()
conflicts = pd.read_csv(DATA / "conflicts.csv")
settings = pd.read_csv(DATA / "settings.csv").iloc[0]

# @block sets
P = products.index.tolist()
T = periods.index.tolist()
CONFLICTS = list(conflicts.itertuples(index=False, name=None))
prev = dict(zip(T[1:], T[:-1]))  # previous week
nxt = dict(zip(T[:-1], T[1:]))  # next week

# @block params
hours = periods["hours"].to_dict()
rate = products["rate"].to_dict()
min_lot = products["min_lot"].to_dict()
max_lot = products["max_lot"].to_dict()
make_cost = products["make_cost"].to_dict()
hold_cost = products["hold_cost"].to_dict()
init_inv = products["init_inv"].to_dict()
max_inv = products["max_inv"].to_dict()
reorder = products["reorder"].to_dict()
max_products = int(settings["max_products"])
smooth_cost = float(settings["smooth_cost"])
peak_cost = float(settings["peak_cost"])

# @block vars
prob = pulp.LpProblem("production_rules", pulp.LpMinimize)
make = pulp.LpVariable.dicts("make", (P, T), lowBound=0)
inv = {p: {t: pulp.LpVariable(f"inv_{p}_{t}", 0, max_inv[p]) for t in T} for p in P}
run = pulp.LpVariable.dicts("run", (P, T), cat="Binary")  # 1 if product p runs in week t

# @block objective
# No abs() or max(): write both as linear epigraphs with helper variables
change = pulp.LpVariable.dicts("change", (P, T[1:]), lowBound=0)
peak = pulp.LpVariable("peak", lowBound=0)
for p in P:
    for t in T[1:]:
        prob += change[p][t] >= make[p][t] - make[p][prev[t]]
        prob += change[p][t] >= make[p][prev[t]] - make[p][t]
for t in T:
    prob += peak >= pulp.lpSum(make[p][t] / rate[p] for p in P)
prob += (
    pulp.lpSum(make_cost[p] * make[p][t] + hold_cost[p] * inv[p][t] for p in P for t in T)
    + smooth_cost * pulp.lpSum(change[p][t] for p in P for t in T[1:])
    + peak_cost * peak
)

# @block constraints
for p in P:
    for t in T:
        prob += inv[p][t] == (init_inv[p] if t == T[0] else inv[p][prev[t]]) + make[p][t] - demand[p, t]
        # Lot size rules are written on the run binaries: run = 0 -> make = 0, run = 1 -> lot size
        prob += make[p][t] >= min_lot[p] * run[p][t]
        prob += make[p][t] <= max_lot[p] * run[p][t]
for t in T:
    prob += pulp.lpSum(make[p][t] / rate[p] for p in P) <= hours[t]
    prob += pulp.lpSum(run[p][t] for p in P) <= max_products
    for p, q in CONFLICTS:
        prob += run[p][t] + run[q][t] <= 1
# Reorder rule, linearized by hand (big-M = reorder point):
# not running next week -> stock at or above reorder point
for p in P:
    for t in T[:-1]:
        prob += inv[p][t] >= reorder[p] * (1 - run[p][nxt[t]])

# @block solve
prob.solve(pulp.HiGHS(msg=False))
objective = pulp.value(prob.objective)
