import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from apps import common_header
from apps import (
    bistro_game,
    global_optimization,
    batch_process,
    facility_location,
    aircrew_training_scheduling,
    risk_return,
    nqueens,
    tips,
    python,
    reformulation_explorer,
)

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


def home():
    st.write("# Welcome to AMPL on Streamlit! 👋")

    st.markdown(
        """
        [AMPL](https://ampl.com) is the most powerful and intuitive tool for developing and deploying
        complex optimization solutions in business & scientific applications.
        """
    )

    st.success(
        "**👈 Select an app from the sidebar** to see some examples of what you can do with AMPL on Streamlit!"
    )

    st.markdown(
        """
        - [📖 Documentation](https://dev.ampl.com/)
        - [💬 Support Forum](https://discuss.ampl.com/)
        - [🧑‍🤝‍🧑 Model Colaboratory](https://ampl.com/colab/)

        Follow us on [LinkedIn](https://www.linkedin.com/company/ampl) and [Twitter](https://twitter.com/AMPLopt) to get the latest updates from the dev team!
        """
    )

    python.main()

    st.markdown(
        """
    [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tips)]
    [[Build your own App with AMPL & Streamlit](https://dev.ampl.com/ampl/python/streamlit.html)]
    """
    )


def app_page(app, icon, title):
    url_path = title.replace(" ", "_")
    if title == "Home":
        url_path = ""

    def page():
        common_header(url_path)
        app()

    return st.Page(
        page,
        url_path=url_path,
        title=title,
        icon=icon,
        default=url_path == "",
    )


pages = {
    "AMPL Streamlit Apps": [
        app_page(home, "🏠", "Home"),
        app_page(risk_return.main, "📈", "Risk Return"),
        app_page(tips.main, "💡", "Modeling Tips"),
    ],
    "Real applications": [
        app_page(aircrew_training_scheduling.main, "✈️", "Aircrew Training Scheduling"),
        app_page(batch_process.main, "⚙️", "Batch Process Optimization"),
        app_page(facility_location.main, "🏭", "Stochastic Facility Location"),
    ],
    "Puzzles & Games": [
        app_page(bistro_game.main, "🍽️", "Bistro Game"),
        app_page(global_optimization.main, "🎅", "Global Optimization"),
        app_page(nqueens.main, "👑", "N-Queens"),
    ],
    "Tools & Documentation": [
        app_page(python.main, "🐍", "Python Integration"),
        app_page(reformulation_explorer.main, "🔍", "Reformulation Explorer"),
    ],
}

pg = st.navigation(pages)
pg.run()
