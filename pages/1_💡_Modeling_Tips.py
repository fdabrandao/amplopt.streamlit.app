import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import tips, common_header, badge

common_header()
tips.main()
st.sidebar.markdown(badge("Modeling_Tips"))
