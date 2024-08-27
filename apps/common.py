import streamlit as st

MP_SOLVERS = [
    "Gurobi",
    "CPLEX",
    "XPRESS",
    "COPT",
    "MOSEK",
    "HiGHS",
    "CBC",
    "SCIP",
    "GCG",
]
MP_SOLVERS_LINKS = ", ".join(
    [
        f"[{solver}](https://dev.ampl.com/solvers/{solver.lower()}/)"
        for solver in MP_SOLVERS
    ]
)


def solver_selector(mp_only=True, default=None, solvers=None):
    assert mp_only == True
    if solvers is None:
        solvers = MP_SOLVERS
    index = 0
    if default is not None:
        assert default in solvers
        index = solvers.index(default)
    solver = st.selectbox(
        "Pick the solver to use 👇", solvers, index=index, key="solver"
    )
    return solver.lower(), solver
