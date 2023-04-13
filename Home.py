import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from apps import INFO_HEADER, INFO_FOOTER

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Logo
_, col2, _ = st.columns((1, 4, 1))
with col2:
    st.image("static/images/logo-inline-web-v4.png")

st.write("# Welcome to AMPL on Streamlit! 👋")

st.sidebar.success("Select an app above.")

st.markdown(INFO_HEADER)

st.markdown(
    """
**👈 Select a demo from the sidebar** to see some examples
    of what you can do with AMPL on Streamlit!
"""
)

st.markdown(INFO_FOOTER)
