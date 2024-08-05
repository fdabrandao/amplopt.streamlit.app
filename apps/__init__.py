import streamlit as st
import platform


def badge(url_path=""):
    if platform.system() != "Linux":
        return ""
    # return f"[![Hits](https://h.ampl.com/https://amplopt.streamlit.app/{url_path})](https://github.com/fdabrandao/amplopt.streamlit.app)"
    return f"<img src='https://h.ampl.com/https://amplopt.streamlit.app/{url_path}' style='float:right;'>"


def common_header(url_path):
    # Logo
    st.image("https://portal.ampl.com/dl/ads/python_ecosystem_badge.png")

    st.write(badge(url_path), unsafe_allow_html=True)
