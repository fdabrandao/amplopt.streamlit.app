# @block setup
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo

DATA = Path(__file__).parent / "data"

# @block data
products = pd.read_csv(DATA / "products.csv", index_col="product")
periods = pd.read_csv(DATA / "periods.csv", index_col="period")
demand = pd.read_csv(DATA / "demand.csv", index_col=["product", "period"])["demand"].to_dict()
conflicts = pd.read_csv(DATA / "conflicts.csv")
discounts = pd.read_csv(DATA / "discounts.csv", index_col="tier")
settings = pd.read_csv(DATA / "settings.csv").iloc[0]

# @block sets
m = pyo.ConcreteModel("production_rules")
m.PROD = pyo.Set(initialize=products.index.tolist())
m.PERIODS = pyo.Set(initialize=periods.index.tolist(), ordered=True)
m.CONFLICTS = pyo.Set(initialize=list(conflicts.itertuples(index=False, name=None)), dimen=2)
m.TIERS = pyo.Set(initialize=discounts.index.tolist(), ordered=True)  # volume discount tiers

# @block params
m.demand = pyo.Param(m.PROD, m.PERIODS, initialize=demand)
m.hours = pyo.Param(m.PERIODS, initialize=periods["hours"].to_dict())
m.rate = pyo.Param(m.PROD, initialize=products["rate"].to_dict())
m.min_lot = pyo.Param(m.PROD, initialize=products["min_lot"].to_dict())
m.max_lot = pyo.Param(m.PROD, initialize=products["max_lot"].to_dict())
m.make_cost = pyo.Param(m.PROD, initialize=products["make_cost"].to_dict())
m.hold_cost = pyo.Param(m.PROD, initialize=products["hold_cost"].to_dict())
m.init_inv = pyo.Param(m.PROD, initialize=products["init_inv"].to_dict())
m.max_inv = pyo.Param(m.PROD, initialize=products["max_inv"].to_dict())
m.reorder = pyo.Param(m.PROD, initialize=products["reorder"].to_dict())
m.max_products = pyo.Param(initialize=int(settings["max_products"]))
m.smooth_cost = pyo.Param(initialize=float(settings["smooth_cost"]))
m.peak_cost = pyo.Param(initialize=float(settings["peak_cost"]))
m.from_units = pyo.Param(m.TIERS, initialize=discounts["from_units"].to_dict())  # where the tier starts
m.price_factor = pyo.Param(m.TIERS, initialize=discounts["price_factor"].to_dict())  # fraction of make_cost

# @block vars
m.Make = pyo.Var(m.PROD, m.PERIODS, bounds=lambda m, p, t: (0, m.max_lot[p]))
m.Inv = pyo.Var(m.PROD, m.PERIODS, bounds=lambda m, p, t: (0, m.max_inv[p]))

# @block objective
# abs() and max() are not linearized for MIP solvers: write them as epigraphs
m.Change = pyo.Var(m.PROD, m.PERIODS, within=pyo.NonNegativeReals)
m.Peak = pyo.Var(within=pyo.NonNegativeReals)


@m.Constraint(m.PROD, m.PERIODS)
def Change_Up(m, p, t):
    if t == m.PERIODS.first():
        return pyo.Constraint.Skip
    return m.Change[p, t] >= m.Make[p, t] - m.Make[p, m.PERIODS.prev(t)]


@m.Constraint(m.PROD, m.PERIODS)
def Change_Down(m, p, t):
    if t == m.PERIODS.first():
        return pyo.Constraint.Skip
    return m.Change[p, t] >= m.Make[p, m.PERIODS.prev(t)] - m.Make[p, t]


@m.Constraint(m.PERIODS)
def Peak_Def(m, t):
    return m.Peak >= sum(m.Make[p, t] / m.rate[p] for p in m.PROD)


# Discounted production cost: a Piecewise component links a cost variable to Make. The discount
# makes it concave, so it needs a representation with binaries (pw_repn) chosen by the modeler
m.Lot_Cost = pyo.Var(m.PROD, m.PERIODS, within=pyo.NonNegativeReals)


def breakpoints(m, p, t):
    return [0.0] + [m.from_units[k] for k in m.TIERS] + [m.max_lot[p]]


def discounted_cost(m, p, t, x):
    edges = breakpoints(m, p, t)
    prices = [m.make_cost[p]] + [m.price_factor[k] * m.make_cost[p] for k in m.TIERS]
    return sum(price * max(0.0, min(x, hi) - lo) for price, lo, hi in zip(prices, edges, edges[1:]))


m.Discount = pyo.Piecewise(
    m.PROD,
    m.PERIODS,
    m.Lot_Cost,
    m.Make,
    pw_pts={(p, t): breakpoints(m, p, t) for p in m.PROD for t in m.PERIODS},
    f_rule=discounted_cost,
    pw_constr_type="EQ",
    pw_repn="INC",
)

m.Total_Cost = pyo.Objective(
    expr=sum(m.Lot_Cost[p, t] + m.hold_cost[p] * m.Inv[p, t] for p in m.PROD for t in m.PERIODS)
    + m.smooth_cost * sum(m.Change[p, t] for p in m.PROD for t in m.PERIODS)
    + m.peak_cost * m.Peak,
    sense=pyo.minimize,
)

# @block constraints
@m.Constraint(m.PROD, m.PERIODS)
def Balance(m, p, t):
    before = m.init_inv[p] if t == m.PERIODS.first() else m.Inv[p, m.PERIODS.prev(t)]
    return m.Inv[p, t] == before + m.Make[p, t] - m.demand[p, t]


@m.Constraint(m.PERIODS)
def Capacity(m, t):
    return sum(m.Make[p, t] / m.rate[p] for p in m.PROD) <= m.hours[t]


# Generalized disjunctive programming: each product-week either runs a lot or is idle
@m.Disjunct(m.PROD, m.PERIODS)
def running(d, p, t):
    d.lot_size = pyo.Constraint(expr=pyo.inequality(m.min_lot[p], m.Make[p, t], m.max_lot[p]))


@m.Disjunct(m.PROD, m.PERIODS)
def idle(d, p, t):
    d.no_make = pyo.Constraint(expr=m.Make[p, t] == 0)
    # Reorder rule as its contrapositive: idle this week -> last week's stock >= reorder point
    if t != m.PERIODS.first():
        d.stock_ok = pyo.Constraint(expr=m.Inv[p, m.PERIODS.prev(t)] >= m.reorder[p])


@m.Disjunction(m.PROD, m.PERIODS)
def run_or_idle(m, p, t):
    return [m.running[p, t], m.idle[p, t]]


@m.LogicalConstraint(m.PERIODS)
def Max_Products(m, t):
    return pyo.atmost(m.max_products, [m.running[p, t].indicator_var for p in m.PROD])


@m.LogicalConstraint(m.CONFLICTS, m.PERIODS)
def No_Mix(m, p, q, t):
    return m.idle[p, t].indicator_var.lor(m.idle[q, t].indicator_var)


# @block solve
pyo.TransformationFactory("gdp.bigm").apply_to(m)
pyo.SolverFactory("highs").solve(m)
objective = pyo.value(m.Total_Cost)
