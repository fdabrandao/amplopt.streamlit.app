# @block setup
from pathlib import Path
import pandas as pd
from docplex.mp.model import Model

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
m = Model("diet")
buy = m.continuous_var_dict(FOOD, lb=f_min, ub=f_max, name="buy")

# @block objective
m.minimize(m.sum(cost[j] * buy[j] for j in FOOD))

# @block constraints
for i in NUTR:
    m.add_range(n_min[i], m.sum(amt[i, j] * buy[j] for j in FOOD), n_max[i], rng_name=f"diet_{i}")

# @block solve
solution = m.solve()
objective = solution.objective_value
