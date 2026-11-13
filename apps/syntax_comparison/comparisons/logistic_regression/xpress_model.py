# @block setup
from pathlib import Path
import pandas as pd
import xpress as xp

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
prob = xp.problem("logistic_regression")
w = {f: prob.addVariable(lb=-xp.infinity, name=f"w_{f}") for f in FEATURES}
b = prob.addVariable(lb=-xp.infinity, name="b")

# @block objective
prob.setObjective(
    xp.Sum(
        xp.log(1 + xp.exp(-(2 * y[i] - 1) * (xp.Sum(w[f] * x[i, f] for f in FEATURES) + b)))
        for i in POINTS
    )
    + lam * xp.Sum(w[f] ** 2 for f in FEATURES),
    sense=xp.minimize,
)

# @block solve
prob.optimize()
objective = prob.attributes.objval
