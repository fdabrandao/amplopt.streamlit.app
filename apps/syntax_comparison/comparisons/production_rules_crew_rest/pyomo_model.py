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
settings = pd.read_csv(DATA / "settings.csv").iloc[0]

# @block sets
m = pyo.ConcreteModel("production_rules")
m.PROD = pyo.Set(initialize=products.index.tolist())
m.PERIODS = pyo.Set(initialize=periods.index.tolist(), ordered=True)
m.CONFLICTS = pyo.Set(initialize=list(conflicts.itertuples(index=False, name=None)), dimen=2)

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
m.long_week = pyo.Param(initialize=float(settings["long_week"]))
m.rest_week = pyo.Param(initialize=float(settings["rest_week"]))

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


m.Total_Cost = pyo.Objective(
    expr=sum(m.make_cost[p] * m.Make[p, t] + m.hold_cost[p] * m.Inv[p, t] for p in m.PROD for t in m.PERIODS)
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


# Crew rest: either this week stays within long_week hours, or next week within rest_week
@m.Disjunction(list(m.PERIODS)[:-1])
def Crew_Rest(m, t):
    this_week = sum(m.Make[p, t] / m.rate[p] for p in m.PROD)
    next_week = sum(m.Make[p, m.PERIODS.next(t)] / m.rate[p] for p in m.PROD)
    return [[this_week <= m.long_week], [next_week <= m.rest_week]]


# @block solve
pyo.TransformationFactory("gdp.bigm").apply_to(m)
pyo.SolverFactory("highs").solve(m)
objective = pyo.value(m.Total_Cost)
