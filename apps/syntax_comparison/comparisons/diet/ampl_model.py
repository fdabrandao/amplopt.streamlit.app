# @block setup
from pathlib import Path
import pandas as pd
from amplpy import AMPL

HERE = Path(__file__).parent
DATA = HERE / "data"
ampl = AMPL()
ampl.read(str(HERE / "model.mod"))

# @block data
ampl.set_data(pd.read_csv(DATA / "foods.csv", index_col="food"), "FOOD")
ampl.set_data(pd.read_csv(DATA / "nutrients.csv", index_col="nutrient"), "NUTR")
ampl.param["amt"] = pd.read_csv(DATA / "amounts.csv", index_col=["nutrient", "food"])["amount"]

# @block solve
ampl.option["solver"] = "highs"  # or gurobi, cplex, xpress, cbc, scip, ...
ampl.solve()
objective = ampl.obj["Total_Cost"].value()
