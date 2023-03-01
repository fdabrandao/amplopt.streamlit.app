import streamlit as st
from amplpy import AMPL, modules


def remove_indentation(block):
    lines = block.strip("\n").split("\n")
    indentation = min((len(s) - len(s.lstrip()) for s in lines if s != ""))
    lines = [s[indentation:] for s in lines]
    return "\n".join(lines)


def snippet(key, model, run, data=""):
    model = remove_indentation(model)
    run = remove_indentation(run)
    modules.load()
    ampl = AMPL()
    st.markdown(f"```python\n{model}\n```")
    ampl.eval(model)
    ampl.eval(data)
    if st.button("Run in AMPL", key=f"btn_{key}"):
        output = ampl.get_output(run)
        with st.expander("In AMPL", expanded=True):
            st.markdown(f"```\n{run}\n```\n\n```\n{output}```")
