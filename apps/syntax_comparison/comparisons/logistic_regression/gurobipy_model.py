# @block setup
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, nlfunc

DATA = Path(__file__).parent / "data"

# @block data
df = pd.read_csv(DATA / "applicants.csv", index_col="applicant")
settings = pd.read_csv(DATA / "settings.csv").iloc[0]

# @block sets
FEATURES = ["income", "debt", "tenure"]
POINTS = df.index.tolist()

# @block params
x = df[FEATURES].stack().to_dict()
y = df["approved"].to_dict()
lam = float(settings["lambda"])

# @block vars
m = gp.Model("logistic_regression")
w = m.addVars(FEATURES, lb=-GRB.INFINITY, name="w")
b = m.addVar(lb=-GRB.INFINITY, name="b")

# @block objective
m.setObjective(
    gp.quicksum(
        nlfunc.log(1 + nlfunc.exp(-(2 * y[i] - 1) * (gp.quicksum(w[f] * x[i, f] for f in FEATURES) + b)))
        for i in POINTS
    )
    + lam * gp.quicksum(w[f] * w[f] for f in FEATURES),
    GRB.MINIMIZE,
)

# @block solve
m.optimize()
objective = m.ObjVal
