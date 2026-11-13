# @block setup
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo

DATA = Path(__file__).parent / "data"

# @block data
foods = pd.read_csv(DATA / "foods.csv", index_col="food")
nutrients = pd.read_csv(DATA / "nutrients.csv", index_col="nutrient")
amounts = pd.read_csv(DATA / "amounts.csv", index_col=["nutrient", "food"])

# @block sets
m = pyo.ConcreteModel("diet")
m.NUTR = pyo.Set(initialize=nutrients.index.tolist())
m.FOOD = pyo.Set(initialize=foods.index.tolist())

# @block params
m.cost = pyo.Param(m.FOOD, initialize=foods["cost"].to_dict())
m.f_min = pyo.Param(m.FOOD, initialize=foods["f_min"].to_dict())
m.f_max = pyo.Param(m.FOOD, initialize=foods["f_max"].to_dict())
m.n_min = pyo.Param(m.NUTR, initialize=nutrients["n_min"].to_dict())
m.n_max = pyo.Param(m.NUTR, initialize=nutrients["n_max"].to_dict())
m.amt = pyo.Param(m.NUTR, m.FOOD, initialize=amounts["amount"].to_dict())

# @block vars
m.Buy = pyo.Var(m.FOOD, bounds=lambda m, j: (m.f_min[j], m.f_max[j]))

# @block objective
m.Total_Cost = pyo.Objective(expr=sum(m.cost[j] * m.Buy[j] for j in m.FOOD), sense=pyo.minimize)

# @block constraints
@m.Constraint(m.NUTR)
def Diet(m, i):
    return (m.n_min[i], sum(m.amt[i, j] * m.Buy[j] for j in m.FOOD), m.n_max[i])


# @block solve
pyo.SolverFactory("highs").solve(m)
objective = pyo.value(m.Total_Cost)
