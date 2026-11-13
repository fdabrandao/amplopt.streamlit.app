# @block setup
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo

DATA = Path(__file__).parent / "data"

# @block data
cities = pd.read_csv(DATA / "cities.csv")
products = pd.read_csv(DATA / "products.csv")
links = pd.read_csv(DATA / "links.csv", index_col=["orig", "dest"])
routes = pd.read_csv(DATA / "routes.csv", index_col=["orig", "dest", "product"])
supply_table = pd.read_csv(DATA / "supply.csv", index_col=["city", "product"])

# @block sets
m = pyo.ConcreteModel("network_flow")
m.CITIES = pyo.Set(initialize=cities["city"].tolist())
m.PRODS = pyo.Set(initialize=products["product"].tolist())
m.LINKS = pyo.Set(initialize=links.index.tolist(), within=m.CITIES * m.CITIES)
m.ROUTES = pyo.Set(initialize=routes.index.tolist(), within=m.LINKS * m.PRODS)
inbound, outbound, on_link = defaultdict(list), defaultdict(list), defaultdict(list)
for i, j, p in m.ROUTES:
    inbound[j, p].append((i, j, p))
    outbound[i, p].append((i, j, p))
    on_link[i, j].append((i, j, p))

# @block params
m.supply = pyo.Param(m.CITIES, m.PRODS, initialize=supply_table["supply"].to_dict(), default=0)
m.cost = pyo.Param(m.ROUTES, initialize=routes["cost"].to_dict())
m.capacity = pyo.Param(m.LINKS, initialize=links["capacity"].to_dict())

# @block vars
m.Ship = pyo.Var(m.ROUTES, within=pyo.NonNegativeReals)

# @block objective
m.Total_Cost = pyo.Objective(expr=sum(m.cost[r] * m.Ship[r] for r in m.ROUTES), sense=pyo.minimize)

# @block constraints
@m.Constraint(m.CITIES, m.PRODS)
def Balance(m, k, p):
    return m.supply[k, p] + sum(m.Ship[r] for r in inbound[k, p]) == sum(m.Ship[r] for r in outbound[k, p])


@m.Constraint(m.LINKS)
def Capacity(m, i, j):
    return sum(m.Ship[r] for r in on_link[i, j]) <= m.capacity[i, j]


# @block solve
pyo.SolverFactory("highs").solve(m)
objective = pyo.value(m.Total_Cost)
