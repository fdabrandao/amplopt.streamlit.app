import streamlit as st
import platform


def badge(url_path=""):
    if platform.system() == "Linux":
        badge_html = f"""<img src="https://h.ampl.com/https://amplopt.streamlit.app/{url_path}">"""
    else:
        badge_html = ""
    return f"""
        <div style="float:right;">
        {badge_html}
        <a href="https://ampl.com/colab" target="_blank" style="text-decoration:none">ampl.com/colab</a>&nbsp;|&nbsp;
        <a href="https://ampl.com/mo-book" target="_blank" style="text-decoration:none">ampl.com/mo-book</a>&nbsp;|&nbsp;
        <a href="https://www.linkedin.com/company/ampl" target="_blank" style="text-decoration:none">Follow us on LinkedIn</a>
        </div>
    """


def common_header(url_path):
    # Logo
    st.image("https://portal.ampl.com/dl/ads/python_ecosystem_badge.png")

    st.write(badge(url_path), unsafe_allow_html=True)
