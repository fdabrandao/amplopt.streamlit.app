import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import python, common_header, badge

st.sidebar.markdown(badge("Python_Integration"))
common_header()
python.main()
