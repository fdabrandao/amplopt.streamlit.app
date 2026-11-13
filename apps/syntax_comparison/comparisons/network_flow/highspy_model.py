# @block setup
from collections import defaultdict
from pathlib import Path
import pandas as pd
import highspy

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
h = highspy.Highs()
ship = h.addVariables(ROUTES, name_prefix="ship")

# @block objective
h.setObjective(h.qsum(cost[r] * ship[r] for r in ROUTES), sense=highspy.ObjSense.kMinimize)

# @block constraints
h.addConstrs(
    supply.get((k, p), 0) + h.qsum(ship[r] for r in inbound[k, p]) == h.qsum(ship[r] for r in outbound[k, p])
    for k in CITIES
    for p in PRODS
)
h.addConstrs(h.qsum(ship[r] for r in on_link[i, j]) <= capacity[i, j] for i, j in LINKS)

# @block solve
h.run()
objective = h.getObjectiveValue()
