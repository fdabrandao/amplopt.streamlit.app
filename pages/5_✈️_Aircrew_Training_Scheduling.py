import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import aircrew_training_scheduling, common_header, badge

st.sidebar.markdown(badge("Aircrew_Training_Scheduling"))
common_header()
aircrew_training_scheduling.main()
