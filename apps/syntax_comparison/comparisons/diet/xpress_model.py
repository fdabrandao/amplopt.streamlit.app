# @block setup
from pathlib import Path
import pandas as pd
import xpress as xp

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
prob = xp.problem("diet")
buy = {j: prob.addVariable(lb=f_min[j], ub=f_max[j], name=f"buy_{j}") for j in FOOD}

# @block objective
prob.setObjective(xp.Sum(cost[j] * buy[j] for j in FOOD), sense=xp.minimize)

# @block constraints
prob.addConstraint(
    xp.constraint(body=xp.Sum(amt[i, j] * buy[j] for j in FOOD), lb=n_min[i], ub=n_max[i], name=f"diet_{i}")
    for i in NUTR
)

# @block solve
prob.optimize()
objective = prob.attributes.objval
