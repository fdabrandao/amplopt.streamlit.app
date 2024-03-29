import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import risk_return, common_header, badge

common_header()
risk_return.main()
st.sidebar.markdown(badge("Risk_Return"))
