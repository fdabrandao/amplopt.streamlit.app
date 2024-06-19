import streamlit as st
import platform

INFO_HEADER = """
[AMPL](https://ampl.com) is the most powerful and intuitive tool for developing and deploying
complex optimization solutions in business & scientific applications.
"""

INFO_FOOTER = """
- [Documentation](https://dev.ampl.com/)
- [Support Forum](https://discuss.ampl.com/)
- [Model Colaboratory](https://ampl.com/colab/)

Follow us on [Twitter](https://twitter.com/AMPLopt) and [LinkedIn](https://www.linkedin.com/company/ampl) to get the latest updates from the dev team!
"""


def common_header():
    # Logo
    st.image("https://portal.ampl.com/dl/ads/python_ecosystem_badge.png")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown(INFO_HEADER)

    st.sidebar.header("Resources")
    st.sidebar.markdown(INFO_FOOTER)


def badge(app=""):
    if platform.system() != "Linux":
        return ""
    return f"[![Hits](https://h.ampl.com/https://amplopt.streamlit.app/{app})](https://github.com/fdabrandao/amplopt.streamlit.app)"
