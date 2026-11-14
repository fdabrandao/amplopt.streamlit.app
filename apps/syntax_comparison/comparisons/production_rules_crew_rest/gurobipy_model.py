# @block setup
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

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
m = gp.Model("production_rules")
make = m.addVars(P, T, name="make")
inv = m.addVars(P, T, ub={(p, t): max_inv[p] for p in P for t in T}, name="inv")
run = m.addVars(P, T, vtype=GRB.BINARY, name="run")  # 1 if product p runs in week t

# @block objective
# abs_() and max_() accept variables only, so differences and loads get their own variables
diff = m.addVars(P, T[1:], lb=-GRB.INFINITY, name="diff")
change = m.addVars(P, T[1:], name="change")
load = m.addVars(T, name="load")
peak = m.addVar(name="peak")
m.addConstrs((diff[p, t] == make[p, t] - make[p, prev[t]] for p in P for t in T[1:]), name="diff_def")
m.addConstrs((change[p, t] == gp.abs_(diff[p, t]) for p in P for t in T[1:]), name="change_def")
m.addConstrs((load[t] == gp.quicksum(make[p, t] / rate[p] for p in P) for t in T), name="load_def")
m.addConstr(peak == gp.max_(load), name="peak_def")
m.setObjective(
    gp.quicksum(make_cost[p] * make[p, t] + hold_cost[p] * inv[p, t] for p in P for t in T)
    + smooth_cost * change.sum()
    + peak_cost * peak,
    GRB.MINIMIZE,
)

# @block constraints
m.addConstrs(
    (
        inv[p, t] == (init_inv[p] if t == T[0] else inv[p, prev[t]]) + make[p, t] - demand[p, t]
        for p in P
        for t in T
    ),
    name="balance",
)
m.addConstrs((load[t] <= hours[t] for t in T), name="capacity")
# Lot size rules are written on the run binaries: run = 0 -> make = 0, run = 1 -> lot size
m.addConstrs((make[p, t] >= min_lot[p] * run[p, t] for p in P for t in T), name="min_lot")
m.addConstrs((make[p, t] <= max_lot[p] * run[p, t] for p in P for t in T), name="max_lot")
m.addConstrs((run.sum("*", t) <= max_products for t in T), name="max_products")
m.addConstrs((run[p, t] + run[q, t] <= 1 for p, q in CONFLICTS for t in T), name="no_mix")
# Reorder rule as its contrapositive: not running next week -> stock at or above reorder point
m.addConstrs(
    ((run[p, nxt[t]] == 0) >> (inv[p, t] >= reorder[p]) for p in P for t in T[:-1]),
    name="reorder",
)
# Crew rest: a week over long_week hours must be followed by at most rest_week hours
long = m.addVars(T[:-1], vtype=GRB.BINARY, name="long")  # 1 if week t may run long
m.addConstrs(((long[t] == 0) >> (load[t] <= long_week) for t in T[:-1]), name="not_long")
m.addConstrs(((long[t] == 1) >> (load[nxt[t]] <= rest_week) for t in T[:-1]), name="crew_rest")

# @block solve
m.optimize()
objective = m.ObjVal
