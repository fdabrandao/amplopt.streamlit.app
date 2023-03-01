import streamlit as st
from content import tip1, tip2, tip3, tip4, tip5

tips = [(t.title, t.run) for t in [tip1, tip2, tip3, tip4, tip5]]
tip_titles = [title for title, _ in tips]


def update_params():
    st.experimental_set_query_params(tip=tip_titles.index(st.session_state.title) + 1)


# Logo and Navigation
_, col2, _ = st.columns((1, 4, 1))
with col2:
    st.image("static/images/logo-inline-web-v4.png")
st.markdown("# AMPL Modeling Tips")

query_params = st.experimental_get_query_params()

if query_params:
    if "tip" in query_params and query_params["tip"][0].isnumeric():
        tip = int(query_params["tip"][0])
        tip_index = min(max(tip, 1), len(tip_titles)) - 1
        st.session_state.title = tip_titles[tip_index]

selected_tip = st.selectbox(
    "Pick the tip 👇", tip_titles, key="title", on_change=update_params
)
tip_index = tip_titles.index(selected_tip)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    """
[AMPL](https://ampl.com) is the most powerful and intuitive tool for developing and deploying
complex optimization solutions in business & scientific applications.
AMPL connects to most open-source and commercial solvers and allows you to switch easily between them.

AMPL has APIs for several popular programming languages
(e.g., [Python](https://amplpy.readthedocs.io/), [R](https://rampl.readthedocs.io/), etc.)
and it allows you to only model once in AMPL and interact with it using an API for a language 
you are familiar with.
    """
)

st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [AMPL Website](https://ampl.com)
- [AMPL Documentation](https://dev.ampl.com/)
- [AMPL Model Colaboraty](https://colab.ampl.com/)

[![Hits](https://h.ampl.com/https://ampl-tips.streamlit.app/)](https://github.com/fdabrandao/ampl-tips)
"""
)

title, run = tips[tip_index]
st.markdown(f"## 💡 {title}")
run()
