# @block setup
from collections import defaultdict
from pathlib import Path
import pandas as pd
from docplex.mp.model import Model

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
LINKS = links.index.tolist()
ROUTES = routes.index.tolist()
inbound, outbound, on_link = defaultdict(list), defaultdict(list), defaultdict(list)
for i, j, p in ROUTES:
    inbound[j, p].append((i, j, p))
    outbound[i, p].append((i, j, p))
    on_link[i, j].append((i, j, p))

# @block params
supply = supply_table["supply"].to_dict()  # missing (city, product) pairs mean 0
cost = routes["cost"].to_dict()
capacity = links["capacity"].to_dict()

# @block vars
m = Model("network_flow")
ship = m.continuous_var_dict(ROUTES, name="ship")

# @block objective
m.minimize(m.sum(cost[r] * ship[r] for r in ROUTES))

# @block constraints
m.add_constraints(
    supply.get((k, p), 0) + m.sum(ship[r] for r in inbound[k, p]) == m.sum(ship[r] for r in outbound[k, p])
    for k in CITIES
    for p in PRODS
)
m.add_constraints(m.sum(ship[r] for r in on_link[i, j]) <= capacity[i, j] for i, j in LINKS)

# @block solve
solution = m.solve()
objective = solution.objective_value
