import streamlit as st
import sys
import os
import streamlit.components.v1 as components

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


def redirect_link():
    if os.environ.get("DEPLOYMENT_METHOD", "") != "docker-deployment":
        return
    components.html(
        """
        <script>
        const currentUrl = window.top.location.href;
        const url = new URL(currentUrl);

        const oldPrefix = "http://localhost:8501";
        const newPrefix = "https://amplopt.streamlit.app";

        if (url.origin != newPrefix) {
            const newUrl = newPrefix + url.pathname + url.search + url.hash;
            document.write("<b><p>Please go to the official URL: <a href='" + newUrl+"' target='_blank'>"+newUrl+"</a></p></b>");
        }
        </script>
        """,
        height=50,
    )


from apps import common_header
from apps import (
    bistro_game,
    global_optimization,
    batch_process,
    facility_location,
    supply_chain,
    aircrew_training_scheduling,
    risk_return,
    lmp,
    tracking_error,
    optimal_control,
    nqueens,
    sudoku,
    tips,
    python,
    reformulation_explorer,
    syntax_comparison,
)

st.set_page_config(
    page_title="AMPL on Streamlit Cloud",
    page_icon="./static/images/cropped-favicon-raw-192x192.png",
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

    app_list = ""
    for group in pages:
        app_list += f"- {group}\n"
        for page in pages[group]:
            if page.url_path == "":
                continue
            app_list += f"    - [{page.icon} {page.title}](/{page.url_path})\n"
    st.markdown(app_list)

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


def app_page(app, icon, title, url_path=None, compact_header=False):
    if url_path is None:
        url_path = title.replace(" ", "_")
    if title == "Home":
        url_path = ""

    def page():
        redirect_link()
        common_header(url_path, compact=compact_header)
        app()
        st.markdown(
            "[AMPL Website](https://ampl.com) | [Follow us on LinkedIn](https://www.linkedin.com/company/ampl) | [Documentation](https://dev.ampl.com) | [Colab Notebooks](https://ampl.com/colab) | [MO-Book](https://ampl.com/mo-book)"
        )

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
        app_page(tips.main, "💡", "Modeling Tips"),
    ],
    "Applications": [
        app_page(aircrew_training_scheduling.main, "✈️", "Aircrew Training Scheduling"),
        app_page(batch_process.main, "⚙️", "Batch Process Optimization"),
        app_page(facility_location.main, "🏭", "Stochastic Facility Location"),
        app_page(supply_chain.main, "📦", "Supply Chain Optimization"),
        app_page(lmp.main, "⚡", "Locational Marginal Pricing"),
        app_page(
            tracking_error.main, "📈", "Tracking Error Optimization", "Tracking_Error"
        ),
        app_page(risk_return.main, "💰", "Portfolio Optimization", "Risk_Return"),
        app_page(tips.main_tip7, "🏷️", "Logistic Regression"),
        app_page(optimal_control.main, "🎯", "Optimal Control"),
    ],
    "Puzzles & Games": [
        app_page(nqueens.main, "👑", "N-Queens"),
        app_page(sudoku.main, "🔢", "Sudoku"),
        app_page(bistro_game.main, "🍽️", "Bistro Game"),
        app_page(global_optimization.main, "🎅", "Global Optimization"),
    ],
    "Tools & Documentation": [
        app_page(python.main, "🐍", "Python Integration"),
        app_page(reformulation_explorer.main, "🔍", "Reformulation Explorer"),
        app_page(
            syntax_comparison.main,
            "⚖️",
            "Modeling Syntax Comparison",
            "Syntax_Comparison",
            compact_header=True,
        ),
    ],
}

st.navigation(pages).run()
