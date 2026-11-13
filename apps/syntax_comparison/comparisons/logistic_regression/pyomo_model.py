# @block setup
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo

DATA = Path(__file__).parent / "data"

# @block data
df = pd.read_csv(DATA / "applicants.csv", index_col="applicant")
settings = pd.read_csv(DATA / "settings.csv").iloc[0]

# @block sets
FEATURES = ["income", "debt", "tenure"]
m = pyo.ConcreteModel("logistic_regression")
m.POINTS = pyo.Set(initialize=df.index.tolist())
m.FEATURES = pyo.Set(initialize=FEATURES)

# @block params
m.x = pyo.Param(m.POINTS, m.FEATURES, initialize=df[FEATURES].stack().to_dict())
m.y = pyo.Param(m.POINTS, initialize=df["approved"].to_dict(), within=pyo.Binary)
m.lam = pyo.Param(initialize=float(settings["lambda"]))

# @block vars
m.w = pyo.Var(m.FEATURES)
m.b = pyo.Var()

# @block objective
m.Log_Loss = pyo.Objective(
    expr=sum(
        pyo.log(1 + pyo.exp(-(2 * m.y[i] - 1) * (sum(m.w[f] * m.x[i, f] for f in m.FEATURES) + m.b)))
        for i in m.POINTS
    )
    + m.lam * sum(m.w[f] ** 2 for f in m.FEATURES),
    sense=pyo.minimize,
)

# @block solve
pyo.SolverFactory("ipopt").solve(m)
objective = pyo.value(m.Log_Loss)
