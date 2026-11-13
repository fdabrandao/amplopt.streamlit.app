# @block setup
from pathlib import Path
import pandas as pd
from amplpy import AMPL

HERE = Path(__file__).parent
DATA = HERE / "data"
ampl = AMPL()
ampl.read(str(HERE / "model.mod"))

# @block data
ampl.set["CITIES"] = pd.read_csv(DATA / "cities.csv")["city"]
ampl.set["PRODS"] = pd.read_csv(DATA / "products.csv")["product"]
ampl.set_data(pd.read_csv(DATA / "links.csv", index_col=["orig", "dest"]), "LINKS")
ampl.set_data(pd.read_csv(DATA / "routes.csv", index_col=["orig", "dest", "product"]), "ROUTES")
ampl.set_data(pd.read_csv(DATA / "supply.csv", index_col=["city", "product"]))

# @block solve
ampl.option["solver"] = "highs"  # or gurobi, cplex, xpress, cbc, scip, ...
ampl.solve()
objective = ampl.obj["Total_Cost"].value()
