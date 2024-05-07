import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import reformulation_explorer, common_header, badge

st.sidebar.markdown(badge("Reformulation_Explorer"))
common_header()
reformulation_explorer.main()
