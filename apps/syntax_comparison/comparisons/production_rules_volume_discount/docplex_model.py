# @block setup
from pathlib import Path
import pandas as pd
from docplex.mp.model import Model

DATA = Path(__file__).parent / "data"

# @block data
products = pd.read_csv(DATA / "products.csv", index_col="product")
periods = pd.read_csv(DATA / "periods.csv", index_col="period")
demand = pd.read_csv(DATA / "demand.csv", index_col=["product", "period"])["demand"].to_dict()
conflicts = pd.read_csv(DATA / "conflicts.csv")
discounts = pd.read_csv(DATA / "discounts.csv", index_col="tier")
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
from_units = discounts["from_units"].tolist()  # weekly units where each tier starts
price_factor = discounts["price_factor"].tolist()  # tier price as a fraction of make_cost

# @block vars
m = Model("production_rules")
make = m.continuous_var_matrix(P, T, name="make")
inv = m.continuous_var_matrix(P, T, ub=lambda pt: max_inv[pt[0]], name="inv")
run = m.binary_var_matrix(P, T, name="run")  # 1 if product p runs in week t

# @block objective
# Discounted production cost: one piecewise-linear function per product, given as (slope, until x)
prices = {p: [make_cost[p]] + [f * make_cost[p] for f in price_factor] for p in P}
cost = {p: m.piecewise_as_slopes(list(zip(prices[p], from_units)), prices[p][-1]) for p in P}
m.minimize(
    m.sum(cost[p](make[p, t]) + hold_cost[p] * inv[p, t] for p in P for t in T)
    + smooth_cost * m.sum(m.abs(make[p, t] - make[p, prev[t]]) for p in P for t in T[1:])
    + peak_cost * m.max(m.sum(make[p, t] / rate[p] for p in P) for t in T)
)

# @block constraints
m.add_constraints(
    inv[p, t] == (init_inv[p] if t == T[0] else inv[p, prev[t]]) + make[p, t] - demand[p, t]
    for p in P
    for t in T
)
m.add_constraints(m.sum(make[p, t] / rate[p] for p in P) <= hours[t] for t in T)
# Lot size rules are written on the run binaries: run = 0 -> make = 0, run = 1 -> lot size
m.add_constraints(make[p, t] >= min_lot[p] * run[p, t] for p in P for t in T)
m.add_constraints(make[p, t] <= max_lot[p] * run[p, t] for p in P for t in T)
m.add_constraints(m.sum(run[p, t] for p in P) <= max_products for t in T)
m.add_constraints(run[p, t] + run[q, t] <= 1 for p, q in CONFLICTS for t in T)
# Reorder rule as its contrapositive: not running next week -> stock at or above reorder point
for p in P:
    for t in T[:-1]:
        m.add_indicator(run[p, nxt[t]], inv[p, t] >= reorder[p], active_value=0)

# @block solve
solution = m.solve()
objective = solution.objective_value
