import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps import bistro_game, common_header, badge

st.sidebar.markdown(badge("Bistro_Game"))
common_header()
bistro_game.main()
