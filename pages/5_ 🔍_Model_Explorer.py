import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import model_explorer, common_header, badge

common_header()
model_explorer.main()
st.sidebar.markdown(badge("Python_Integration"))
