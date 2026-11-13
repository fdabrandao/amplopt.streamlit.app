# @block setup
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA = Path(__file__).parent / "data"

# @block data
foods = pd.read_csv(DATA / "foods.csv", index_col="food")
nutrients = pd.read_csv(DATA / "nutrients.csv", index_col="nutrient")
amounts = pd.read_csv(DATA / "amounts.csv", index_col=["nutrient", "food"])

# @block sets
NUTR = nutrients.index.tolist()
FOOD = foods.index.tolist()

# @block params
cost = foods["cost"].to_dict()
f_min = foods["f_min"].to_dict()
f_max = foods["f_max"].to_dict()
n_min = nutrients["n_min"].to_dict()
n_max = nutrients["n_max"].to_dict()
amt = amounts["amount"].to_dict()

# @block vars
m = gp.Model("diet")
buy = m.addVars(FOOD, lb=f_min, ub=f_max, name="buy")

# @block objective
m.setObjective(buy.prod(cost), GRB.MINIMIZE)

# @block constraints
m.addConstrs(
    (gp.quicksum(amt[i, j] * buy[j] for j in FOOD) == [n_min[i], n_max[i]] for i in NUTR),
    name="diet",
)

# @block solve
m.optimize()
objective = m.ObjVal
