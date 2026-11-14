# @block setup
from pathlib import Path
import pandas as pd
import highspy

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
long_week = float(settings["long_week"])
rest_week = float(settings["rest_week"])

# @block vars
h = highspy.Highs()
make = h.addVariables(P, T, name_prefix="make")
inv = h.addVariables(P, T, ub={(p, t): max_inv[p] for p in P for t in T}, name_prefix="inv")
run = h.addBinaries(P, T, name_prefix="run")  # 1 if product p runs in week t

# @block objective
# No abs() or max(): write both as linear epigraphs with helper variables
change = h.addVariables(P, T[1:], name_prefix="change")
peak = h.addVariable(name="peak")
h.addConstrs(change[p, t] >= make[p, t] - make[p, prev[t]] for p in P for t in T[1:])
h.addConstrs(change[p, t] >= make[p, prev[t]] - make[p, t] for p in P for t in T[1:])
h.addConstrs(peak >= h.qsum(make[p, t] / rate[p] for p in P) for t in T)
h.setObjective(
    h.qsum(make_cost[p] * make[p, t] + hold_cost[p] * inv[p, t] for p in P for t in T)
    + smooth_cost * h.qsum(change.values())
    + peak_cost * peak,
    sense=highspy.ObjSense.kMinimize,
)

# @block constraints
h.addConstrs(
    inv[p, t] == (init_inv[p] if t == T[0] else inv[p, prev[t]]) + make[p, t] - demand[p, t]
    for p in P
    for t in T
)
h.addConstrs(h.qsum(make[p, t] / rate[p] for p in P) <= hours[t] for t in T)
# Lot size rules are written on the run binaries: run = 0 -> make = 0, run = 1 -> lot size
h.addConstrs(make[p, t] >= min_lot[p] * run[p, t] for p in P for t in T)
h.addConstrs(make[p, t] <= max_lot[p] * run[p, t] for p in P for t in T)
h.addConstrs(h.qsum(run[p, t] for p in P) <= max_products for t in T)
h.addConstrs(run[p, t] + run[q, t] <= 1 for p, q in CONFLICTS for t in T)
# Reorder rule, linearized by hand (big-M = reorder point):
# not running next week -> stock at or above reorder point
h.addConstrs(inv[p, t] >= reorder[p] * (1 - run[p, nxt[t]]) for p in P for t in T[:-1])
# Crew rest, linearized by hand (big-M = line hours above each limit):
# a week over long_week hours must be followed by at most rest_week hours
long = h.addBinaries(T[:-1], name_prefix="long")  # 1 if week t may run long
h.addConstrs(
    h.qsum(make[p, t] / rate[p] for p in P) <= long_week + max(0, hours[t] - long_week) * long[t]
    for t in T[:-1]
)
h.addConstrs(
    h.qsum(make[p, nxt[t]] / rate[p] for p in P)
    <= rest_week + max(0, hours[nxt[t]] - rest_week) * (1 - long[t])
    for t in T[:-1]
)

# @block solve
h.run()
objective = h.getObjectiveValue()
