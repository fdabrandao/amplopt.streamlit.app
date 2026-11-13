# @block setup
from pathlib import Path
import pandas as pd
import highspy

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
h = highspy.Highs()
buy = h.addVariables(FOOD, lb=f_min, ub=f_max, name_prefix="buy")

# @block objective
h.setObjective(h.qsum(cost[j] * buy[j] for j in FOOD), sense=highspy.ObjSense.kMinimize)

# @block constraints
h.addConstrs(n_min[i] <= h.qsum(amt[i, j] * buy[j] for j in FOOD) <= n_max[i] for i in NUTR)

# @block solve
h.run()
objective = h.getObjectiveValue()
