# @block setup
from pathlib import Path
import pandas as pd
from amplpy import AMPL

HERE = Path(__file__).parent
DATA = HERE / "data"
ampl = AMPL()
ampl.read(str(HERE / "model.mod"))

# @block data
ampl.set_data(pd.read_csv(DATA / "products.csv", index_col="product"), "PROD")
ampl.set_data(pd.read_csv(DATA / "periods.csv", index_col="period"), "PERIODS")
ampl.set_data(pd.read_csv(DATA / "demand.csv", index_col=["product", "period"]))
ampl.set["CONFLICTS"] = pd.read_csv(DATA / "conflicts.csv").itertuples(index=False)
ampl.set_data(pd.read_csv(DATA / "discounts.csv", index_col="tier"), "TIERS")
for name, value in pd.read_csv(DATA / "settings.csv").iloc[0].items():
    ampl.param[name] = value

# @block solve
ampl.option["solver"] = "highs"  # or gurobi, cplex, xpress, cbc, scip, ...
ampl.solve()
objective = ampl.obj["Total_Cost"].value()
