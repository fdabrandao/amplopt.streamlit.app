# @block setup
from pathlib import Path
import pandas as pd
import xpress as xp

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
prob = xp.problem("production_rules")
make = prob.addVariables(P, T, name="make")
inv = {(p, t): prob.addVariable(ub=max_inv[p], name=f"inv_{p}_{t}") for p in P for t in T}
run = prob.addVariables(P, T, vartype=xp.binary, name="run")  # 1 if product p runs in week t

# @block constraints
prob.addConstraint(
    inv[p, t] == (init_inv[p] if t == T[0] else inv[p, prev[t]]) + make[p, t] - demand[p, t]
    for p in P
    for t in T
)
prob.addConstraint(xp.Sum(make[p, t] / rate[p] for p in P) <= hours[t] for t in T)
# Lot size rules are written on the run binaries: run = 0 -> make = 0, run = 1 -> lot size
prob.addConstraint(make[p, t] >= min_lot[p] * run[p, t] for p in P for t in T)
prob.addConstraint(make[p, t] <= max_lot[p] * run[p, t] for p in P for t in T)
prob.addConstraint(xp.Sum(run[p, t] for p in P) <= max_products for t in T)
prob.addConstraint(run[p, t] + run[q, t] <= 1 for p, q in CONFLICTS for t in T)
# Reorder rule as its contrapositive: not running next week -> stock at or above reorder point
prob.addIndicator(
    *[(run[p, nxt[t]] == 0, inv[p, t] >= reorder[p]) for p in P for t in T[:-1]]
)

# @block objective
prob.setObjective(
    xp.Sum(make_cost[p] * make[p, t] + hold_cost[p] * inv[p, t] for p in P for t in T)
    + smooth_cost * xp.Sum(xp.abs(make[p, t] - make[p, prev[t]]) for p in P for t in T[1:])
    + peak_cost * xp.max([xp.Sum(make[p, t] / rate[p] for p in P) for t in T]),
    sense=xp.minimize,
)

# @block solve
prob.optimize()
objective = prob.attributes.objval
