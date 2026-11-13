# @block setup
from pathlib import Path
import pandas as pd
import pulp

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
prob = pulp.LpProblem("diet", pulp.LpMinimize)
buy = {j: pulp.LpVariable(f"buy_{j}", f_min[j], f_max[j]) for j in FOOD}

# @block objective
prob += pulp.lpSum(cost[j] * buy[j] for j in FOOD)

# @block constraints
for i in NUTR:
    intake = pulp.lpSum(amt[i, j] * buy[j] for j in FOOD)
    prob += intake >= n_min[i], f"diet_min_{i}"
    prob += intake <= n_max[i], f"diet_max_{i}"

# @block solve
prob.solve(pulp.HiGHS(msg=False))
objective = pulp.value(prob.objective)
