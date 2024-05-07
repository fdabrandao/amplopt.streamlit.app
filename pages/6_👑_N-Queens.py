import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import nqueens, common_header, badge

st.sidebar.markdown(badge("N-Queens"))
common_header()
nqueens.main()
