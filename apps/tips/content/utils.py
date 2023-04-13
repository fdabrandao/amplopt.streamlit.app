import os
import streamlit as st
from amplpy import AMPL

MPSOLVERS = ["highs", "cbc", "gurobi", "xpress", "copt"]


def remove_indentation(block):
    if not block:
        return block
    lines = block.strip("\n").split("\n")
    indentation = min((len(s) - len(s.lstrip()) for s in lines if s != ""))
    lines = [s[indentation:] for s in lines]
    return "\n".join(lines)


def snippet(key, model, run, data="", data_code="", solvers=None):
    model = remove_indentation(model)
    run = remove_indentation(run)
    data_code = remove_indentation(data_code)
    ampl = AMPL()
    ampl.eval(model)
    ampl.eval(data)
    st.markdown(f"```python\n{model}\n```")
    if data_code:
        st.markdown(f"```python\n{data_code}\n```")
        exec(data_code, globals(), locals())
    if solvers is not None:
        selected_solver = st.selectbox(
            "Pick the solver 👇", solvers, key=f"solver_{key}"
        )
        run = run.replace("$SOLVER", selected_solver)
    if st.button("Run in AMPL", key=f"btn_{key}"):
        output = ampl.get_output(run)
        with st.expander("In AMPL", expanded=True):
            st.markdown(f"```\n{run}\n```\n\n```\n{output}```")
