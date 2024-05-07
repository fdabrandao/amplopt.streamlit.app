import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import facility_location, common_header, badge

st.sidebar.markdown(badge("Stochastic_Facility_Location"))
common_header()
facility_location.main()
