import streamlit as st
from .content import tip1, tip2, tip3, tip4, tip5, tip6, tip7, tip8, tip9, tip10, tip11


def main():
    tips = [
        (t.title, t.run)
        for t in [tip1, tip2, tip3, tip4, tip5, tip6, tip7, tip8, tip9, tip10, tip11]
    ]
    tip_titles = [title.replace("`", "") for title, _ in tips]

    def update_params():
        st.query_params["tip"] = tip_titles.index(st.session_state.title) + 1

    st.title("💡 AMPL Modeling Tips")

    tip_index = len(tip_titles) - 1
    if "tip" in st.query_params and st.query_params["tip"].isnumeric():
        tip = int(st.query_params["tip"])
        tip_index = min(max(tip, 1), len(tip_titles)) - 1
    st.session_state.title = tip_titles[tip_index]

    selected_tip = st.selectbox(
        "Pick the tip 👇", tip_titles, key="title", on_change=update_params
    )
    tip_index = tip_titles.index(selected_tip)

    title, run = tips[tip_index]
    st.markdown(f"## 💡 {title}")
    run()


def main_tip7():
    st.title(f"🎯 {tip7.title[tip7.title.find(':')+1:].strip()}")
    tip7.run()
