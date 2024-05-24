import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import batch_process, common_header, badge

st.sidebar.markdown(badge("Batch_Process_Optimization"))
common_header()
batch_process.main()
