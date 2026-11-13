# @block setup
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA = Path(__file__).parent / "data"

# @block data
cities = pd.read_csv(DATA / "cities.csv")
products = pd.read_csv(DATA / "products.csv")
links = pd.read_csv(DATA / "links.csv", index_col=["orig", "dest"])
routes = pd.read_csv(DATA / "routes.csv", index_col=["orig", "dest", "product"])
supply_table = pd.read_csv(DATA / "supply.csv", index_col=["city", "product"])

# @block sets
CITIES = cities["city"].tolist()
PRODS = products["product"].tolist()
LINKS = gp.tuplelist(links.index)
ROUTES = gp.tuplelist(routes.index)

# @block params
supply = supply_table["supply"].to_dict()  # missing (city, product) pairs mean 0
cost = routes["cost"].to_dict()
capacity = links["capacity"].to_dict()

# @block vars
m = gp.Model("network_flow")
ship = m.addVars(ROUTES, name="ship")

# @block objective
m.setObjective(ship.prod(cost), GRB.MINIMIZE)

# @block constraints
m.addConstrs(
    (supply.get((k, p), 0) + ship.sum("*", k, p) == ship.sum(k, "*", p) for k in CITIES for p in PRODS),
    name="balance",
)
m.addConstrs((ship.sum(i, j, "*") <= capacity[i, j] for i, j in LINKS), name="capacity")

# @block solve
m.optimize()
objective = m.ObjVal
