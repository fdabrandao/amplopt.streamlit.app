import streamlit as st
from amplpy import AMPL
import os
from ..common import solver_selector
from .data import InputData
from .reports import Reports
from .model import ModelBuilder


def main():
    st.title("📦 Supply Chain Optimization")

    def require_rerun():
        st.session_state["needs_rerun"] = True

    options = [
        "Homework 1: Demand Balance + Inventory Carryover + Material Balance",
        "Homework 2: Shelf-Life + Production Hours + Resource Capacity",
        "Homework 3: Transfers+ Target Stocks + Storage Capacity",
    ]

    if "homework" not in st.query_params:
        st.query_params["homework"] = 3

    default_option = max(0, int(st.query_params["homework"]) - 1)

    def update_params():
        if "homework" in st.session_state:
            st.query_params["homework"] = (
                options.index(st.session_state["homework"]) + 1
            )

    class_number = (
        options.index(
            st.selectbox(
                "Production Optimization Class",
                options,
                key="homework",
                index=default_option,
                on_change=update_params(),
            )
        )
        + 1
    )

    st.session_state.instance = InputData(
        os.path.join(os.path.dirname(__file__), "InputDataProductionSolver.xlsx"),
        class_number,
        on_change=require_rerun,
    )
    instance = st.session_state.instance

    st.markdown("## Production Optimization")

    col1, col2 = st.columns(2)
    with col1:
        use_restrict_table = st.checkbox(
            "Use restrict table Product x Locations", value=True
        )
    with col2:
        show_complete_model = st.checkbox("Show complete model", value=False)

    if class_number == 2:
        with col1:
            model_shelf_life = st.checkbox("Model shelf-life", value=True)
    else:
        model_shelf_life = False

    st.session_state.mb = ModelBuilder(
        class_number=class_number,
        use_restrict_table=use_restrict_table,
        show_complete_model=show_complete_model,
        model_shelf_life=model_shelf_life,
        on_change=require_rerun,
    )
    mb = st.session_state.mb

    st.code(mb.model)

    ampl = AMPL()
    ampl.eval(mb.model)

    if show_complete_model:
        pass
    elif class_number == 1:
        st.markdown("## 🧑‍🏫 Exercises")
        exercises = [
            "",
            "All",
            "Exercise #1: Demand Balance",
            "Exercise #2: Inventory Carryover",
            "Exercise #3: Material Balance",
        ]
        selected_exercise = (
            exercises.index(
                st.selectbox(
                    "Select the exercise(s) you want to complete 👇",
                    exercises,
                    key="exercise",
                    index=0,
                    on_change=require_rerun,
                )
            )
            - 1
        )
        mb.demand_fulfillment_exercise(
            ampl, exercise=1, selected_exercise=selected_exercise
        )
        mb.inventory_carryover_exercise(
            ampl, exercise=2, selected_exercise=selected_exercise
        )
        mb.material_balance_exercise(
            ampl, exercise=3, selected_exercise=selected_exercise
        )
    elif class_number == 2:
        st.markdown("## 🧑‍🏫 Exercises")
        exercises = [
            "",
            "All",
            "Exercise #1: Production Hours",
            "Exercise #2: Resource Capacity",
        ]
        selected_exercise = (
            exercises.index(
                st.selectbox(
                    "Select the exercise(s) you want to complete 👇",
                    exercises,
                    key="exercise",
                    index=0,
                    on_change=require_rerun,
                )
            )
            - 1
        )
        mb.production_rate_exercise(
            ampl, exercise=1, selected_exercise=selected_exercise
        )
        mb.resource_capacity_exercise(
            ampl, exercise=2, selected_exercise=selected_exercise
        )
    elif class_number == 3:
        st.markdown("## 🧑‍🏫 Exercises")
        exercises = [
            "",
            "All",
            "Exercise #1: Transfers",
            "Exercise #2: Target Stocks",
            "Exercise #3: Storage Capacity",
        ]
        selected_exercise = (
            exercises.index(
                st.selectbox(
                    "Select the exercise(s) you want to complete 👇",
                    exercises,
                    key="exercise",
                    index=0,
                    on_change=require_rerun,
                )
            )
            - 1
        )
        mb.material_balance_with_transfers_exercise(
            ampl, exercise=1, selected_exercise=selected_exercise
        )
        mb.target_stock_exercise(ampl, exercise=2, selected_exercise=selected_exercise)
        mb.storage_capacity_exercise(
            ampl, exercise=3, selected_exercise=selected_exercise
        )

    st.markdown("## Solve")

    with st.expander("Dimensions"):
        instance.filter_dimensions()

    with st.expander("Data"):
        instance.edit_data()

    demand = instance.demand[["Product", "Location", "Period", "Quantity"]].copy()
    starting_inventory = instance.starting_inventory[
        ["Product", "Location", "Quantity"]
    ].copy()
    demand["Period"] = demand["Period"].dt.strftime("%Y-%m-%d")
    periods = list(sorted(set(demand["Period"])))
    demand.set_index(["Product", "Location", "Period"], inplace=True)
    starting_inventory.set_index(["Product", "Location"], inplace=True)

    try:
        ampl.set["PRODUCTS"] = instance.selected_products
        ampl.set["LOCATIONS"] = instance.selected_locations
        ampl.set["PRODUCTS_LOCATIONS"] = instance.products_locations
        ampl.set["PERIODS"] = periods
        ampl.param["Demand"] = demand["Quantity"]
        ampl.param["InitialInventory"] = starting_inventory["Quantity"]

        if class_number >= 2:
            ampl.set["RESOURCES"] = instance.all_resources
            ampl.param["ProductionRate"] = instance.production_rate.set_index(
                ["Product", "Location", "Resource"]
            )[["Rate"]]
            ampl.param["AvailableCapacity"] = instance.available_capacity.set_index(
                ["Resource", "Location"]
            )

        if class_number >= 3:
            ampl.set["TRANSFER_LANES"] = list(
                instance.transfer_lanes.itertuples(index=False, name=None)
            )
            ampl.param["TargetStock"] = instance.target_stocks.set_index(
                ["Product", "Location"]
            )
            ampl.param["MaxCapacity"] = instance.location_capacity.set_index(
                ["Location"]
            )

    except Exception as e:
        message = str(e)
        if message.startswith('Error executing "let" command:'):
            message = message[message.find(":") + 1 :].strip()
            st.error(f"Error setting data: {message}")
            st.stop()
        else:
            pass

    with st.expander("Adjust parameters"):
        col1, col2 = st.columns(2)
        with col1:
            ampl.param["UnmetDemandPenalty"] = st.slider(
                "UnmetDemandPenalty:",
                min_value=0,
                max_value=50,
                value=10,
                on_change=require_rerun,
            )

            if model_shelf_life:
                ampl.param["MaxShelfLife"] = st.slider(
                    "MaxShelfLife:",
                    min_value=0,
                    max_value=5,
                    value=3,
                    on_change=require_rerun,
                )

                ampl.param["EnsureOldStockGoesFirst"] = st.checkbox(
                    "Sell old inventory first", value=True
                )

        with col2:
            ampl.param["EndingInventoryPenalty"] = st.slider(
                "EndingInventoryPenalty:",
                min_value=0,
                max_value=50,
                value=5,
                on_change=require_rerun,
            )

            if model_shelf_life:
                ampl.param["LostInventoryPenalty"] = st.slider(
                    "LostInventoryPenalty:",
                    min_value=0,
                    max_value=50,
                    value=10,
                    on_change=require_rerun,
                )

        if class_number >= 3:
            with col1:
                ampl.param["AboveTargetPenalty"] = st.slider(
                    "AboveTargetPenalty:",
                    min_value=0,
                    max_value=50,
                    value=2,
                    on_change=require_rerun,
                )

            with col2:
                ampl.param["BelowTargetPenalty"] = st.slider(
                    "BelowTargetPenalty:",
                    min_value=0,
                    max_value=50,
                    value=3,
                    on_change=require_rerun,
                )

            with col1:
                ampl.param["TransferPenalty"] = st.slider(
                    "TransferPenalty:",
                    min_value=0,
                    max_value=50,
                    value=1,
                    on_change=require_rerun,
                )

    auto_rerun = st.checkbox(
        "Automatically rerun the solve process to update the results", value=True
    )

    if (
        auto_rerun
        or not st.session_state.get("needs_rerun", False)
        or st.button("Rerun the solve process to update the results", type="primary")
    ):
        st.session_state["needs_rerun"] = False
        # Select the solver to use
        solver, _ = solver_selector(mp_only=True)
        # Solve the problem
        output = ampl.solve(solver=solver, mp_options="outlev=1", return_output=True)
        if ampl.solve_result != "solved":
            st.error(f"The model could not be solved:\n```\n{output}\n```")
        else:
            with st.expander("Solver Output", expanded=True):
                st.write(f"```\n{output}\n```")

        if ampl.solve_result == "solved":
            ampl.option["display_width"] = 1000
            model = ampl.export_model()
            model = model[: model.find("###model-end")] + "###model-end"

            st.markdown(
                "Download the model, data, or a complete session snapshot to run elsewhere 👇"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 Download Model",
                    data=model,
                    file_name="prodopt.mod",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    label="📥 Download Data",
                    data=ampl.export_data(),
                    file_name="prodopt.dat",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col3:
                st.download_button(
                    label="📥 Download Snapshot",
                    help="Download a run file that allows reproducing the session state elsewhere",
                    data=ampl.snapshot(),
                    file_name="session.run",
                    mime="text/plain",
                    use_container_width=True,
                )

            # Reports
            st.markdown("## Reports")
            reports = Reports(instance, ampl)

            st.markdown("### Demand Report")
            reports.demand_report()

            st.markdown("### Material Balance Report")
            reports.material_balance_report(
                include_shelf_life=model_shelf_life,
                include_transfers=class_number >= 3,
                include_target_stock=class_number >= 3,
            )

            if class_number >= 2:
                st.markdown("### Resource Utilization Report")
                reports.resource_utilization_report()

    st.markdown(
        """##### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/supply_chain)] [[ChatGPT Solving Homework exercises]](https://chatgpt.com/share/e6f49ec8-3931-4586-b944-f104aebacd46)"""
    )
