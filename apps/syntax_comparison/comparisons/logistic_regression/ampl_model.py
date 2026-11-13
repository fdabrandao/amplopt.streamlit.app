# @block setup
from pathlib import Path
import pandas as pd
from amplpy import AMPL

HERE = Path(__file__).parent
DATA = HERE / "data"
ampl = AMPL()
ampl.read(str(HERE / "model.mod"))

# @block data
df = pd.read_csv(DATA / "applicants.csv", index_col="applicant")
features = ["income", "debt", "tenure"]
ampl.set["POINTS"] = df.index
ampl.set["FEATURES"] = features
ampl.param["x"] = df[features].stack()
ampl.param["y"] = df["approved"]
ampl.param["lambda"] = pd.read_csv(DATA / "settings.csv")["lambda"].iloc[0]

# @block solve
ampl.option["solver"] = "ipopt"  # or knitro, gurobi, ...
ampl.solve()
objective = ampl.obj["Log_Loss"].value()
