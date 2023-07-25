import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from apps import badge, python

st.set_page_config(
    page_title="AMPL on Streamlit Cloud",
    page_icon="👋",
)


@st.cache_data
def activate_license():
    from amplpy import modules

    # Activate the license (e.g., a free https://ampl.com/ce license)
    uuid = os.environ.get("AMPLKEY_UUID", None)
    if uuid is not None:
        modules.activate(uuid)
    return uuid


activate_license()


# Banner
st.image("https://portal.ampl.com/dl/ads/python_ecosystem_badge.png")

st.write("# Welcome to AMPL on Streamlit! 👋")

st.sidebar.success("Select an app above.")

st.markdown(
    """
[AMPL](https://ampl.com) is the most powerful and intuitive tool for developing and deploying
complex optimization solutions in business & scientific applications.
            
**👈 Select a demo from the sidebar** to see some examples of what you can do with AMPL on Streamlit!"""
)

links = {}
for fname in os.listdir("pages"):
    if fname.endswith(".py"):
        fname = fname[:-3]
        page_number = int(fname[: fname.find("_")])
        fname = fname[fname.find("_") + 1 :]
        url = fname[fname.find("_") + 1 :]
        label = fname.replace("_", " ")
        links[page_number] = f"[{label}]({url})"
for page_number in sorted(links):
    st.markdown(links[page_number])

python.main()

st.markdown(badge(""))
st.markdown(
    """
[[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tips)]
[[Build your own App with AMPL & Streamlit](https://dev.ampl.com/ampl/python/streamlit.html)]
"""
)
