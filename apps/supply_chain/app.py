import streamlit as st
from amplpy import AMPL
import os
from ..common import solver_selector
from .data import InputData
from .reports import Reports
from .model import ModelBuilder


def main():
    st.title("📦 Supply Chain Optimization")

    st.markdown(
        """
        During our [hands-on masterclass on Supply Chain](https://dev.ampl.com/ampl/videos/supply-chain.html) we show how to build a powerful network optimization solver from scratch.
        We begin with creating a production planning solver and progressively expand it into a full network optimization solution.
        """
    )

    def require_rerun():
        st.session_state["needs_rerun"] = True

    problems = [
        "Part 1: Production Optimization: The demand has a location (item/location)",
        "Part 2: Network Optimization: The Demand has no location assigned (item/customer)",
    ]

    if "problem" not in st.query_params:
        st.query_params["problem"] = len(problems)

    default_problem = min(
        max(0, int(st.query_params.get("problem", 0)) - 1), len(problems) - 1
    )

    def update_problem_param():
        if "problem" in st.session_state:
            st.query_params["problem"] = problems.index(st.session_state["problem"]) + 1

    problem_number = (
        problems.index(
            st.selectbox(
                "Select the Supply Chain Optimization Problem 👇",
                problems,
                key="problem",
                index=default_problem,
                on_change=update_problem_param(),
            )
        )
        + 1
    )

    homeworks = [
        "Homework 1: Demand Balance + Inventory Carryover + Material Balance",
        "Homework 2: Shelf-Life + Production Hours + Resource Capacity",
        "Homework 3: Transfers + Target Stocks + Storage Capacity",
        "Homework 4: Min and Min+Incremental Lot-sizing",
    ]

    if "homework" not in st.query_params:
        st.query_params["homework"] = len(homeworks)

    default_homework = min(
        max(0, int(st.query_params.get("homework", 0)) - 1), len(homeworks) - 1
    )

    def update_homework_param():
        if "homework" in st.session_state:
            st.query_params["homework"] = (
                homeworks.index(st.session_state["homework"]) + 1
            )

    homework_number = (
        homeworks.index(
            st.selectbox(
                "Select the Supply Chain Optimization homework 👇",
                homeworks,
                key="homework",
                index=default_homework,
                on_change=update_homework_param(),
            )
        )
        + 1
    )

    st.session_state.instance = InputData(
        os.path.join(os.path.dirname(__file__), "InputDataProductionSolver.xlsx"),
        problem_number=problem_number,
        homework_number=homework_number,
        on_change=require_rerun,
    )
    instance = st.session_state.instance

    # Initialize the scenario list
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    if "last_scenario" not in st.session_state:
        st.session_state.last_scenario = None

    if problem_number == 1:
        st.markdown("## Production Optimization")
    else:
        st.markdown("## Network Optimization")

    col1, col2 = st.columns(2)
    with col1:
        use_restrict_table = st.checkbox("Use restrict tables", value=True)
    with col2:
        show_complete_model = st.checkbox("Show complete model", value=True)

    model_shelf_life = False
    if homework_number == 2:
        with col1:
            model_shelf_life = st.checkbox("Model shelf-life", value=True)

    layered_targets = False
    layered_storage_capacity = False
    if homework_number == 3:
        with col1:
            layered_storage_capacity = st.checkbox(
                "Layered Max Storage Capacity", value=False
            )
            layered_targets = st.checkbox("Layered Targets", value=False)
    lot_sizing_mp, model_incremental_lot_sizing = False, False

    include_homework2 = homework_number >= 2
    include_homework3 = homework_number >= 3

    if homework_number == 4:
        include_homework2 = st.checkbox(
            "Model Production Hours + Resource Capacity (from Homework 2)",
            value=True,
        )
        include_homework3 = st.checkbox(
            "Model Transfers + Target Stocks + Storage Capacity (from Homework 3)",
            value=True,
        )
        with col1:
            homeworks = [
                "Min Lot-Sizing",
                "Min+Incremental Lot-Sizing",
            ]
            choice = st.radio("Type of lot-sizing?", homeworks)
            model_incremental_lot_sizing = homeworks.index(choice) == 1

            homeworks = [
                "High-Level Logic Modeling (via AMPL MP)",
                "Old-School Big-M Method",
            ]
            choice = st.radio("How to model lot-sizing?", homeworks)
            lot_sizing_mp = homeworks.index(choice) == 0

    st.session_state.mb = ModelBuilder(
        problem_number=problem_number,
        homework_number=homework_number,
        use_restrict_table=use_restrict_table,
        show_complete_model=show_complete_model,
        model_shelf_life=model_shelf_life,
        layered_storage_capacity=layered_storage_capacity,
        layered_targets=layered_targets,
        model_incremental_lot_sizing=model_incremental_lot_sizing,
        lot_sizing_mp=lot_sizing_mp,
        include_homework2=include_homework2,
        include_homework3=include_homework3,
        on_change=require_rerun,
    )
    mb = st.session_state.mb

    st.code(mb.model)

    ampl = AMPL()
    ampl.eval(mb.model)
    if not show_complete_model:
        mb.display_exercises(ampl=ampl)

    st.markdown("## Solve")

    with st.expander("Dimensions"):
        instance.filter_dimensions()

    with st.expander("Data"):
        instance.edit_data()
        # Load data into AMPL
        instance.load_data(ampl=ampl)

    # Adjust model parameters
    mb.adjust_parameters(ampl=ampl)

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
        mp_options = "outlev=1 timelim=4 cvt:bigM=1e6"
        if problem_number == 2:
            mp_options += " multiobj=1"
        output = ampl.solve(solver=solver, mp_options=mp_options, return_output=True)

        if ampl.solve_result not in ["solved", "limit"]:
            st.error(f"The model could not be solved:\n```\n{output}\n```")
        else:
            with st.expander("Solver Output", expanded=True):
                st.write(f"```\n{output}\n```")

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

                st.session_state.last_scenario = {
                    "name": None,
                    "snapshot": ampl.snapshot(),
                    "output": output,
                }

                # Allow saving the scenario
                scenario_name = st.text_input(
                    "Enter a name for the scenario:",
                    value=f"Scenario #{len(st.session_state.scenarios)+1}",
                )
                if st.button("Save Scenario", type="primary"):
                    if scenario_name:
                        st.session_state.scenarios.append(
                            {
                                "name": scenario_name,
                                "snapshot": st.session_state.last_scenario["snapshot"],
                                "output": st.session_state.last_scenario["output"],
                            }
                        )
                        st.success("Scenario saved!")
                    else:
                        st.warning("Please enter a scenario name before saving.")

            st.markdown("## Scenarios")

            def select_scenario(label="", key=""):
                scenario = None
                options = [""] + [s["name"] for s in st.session_state.scenarios]
                index = (
                    options.index(
                        st.selectbox(
                            label or f"Select the scenario to see 👇",
                            options,
                            key=f"{key}_scenario",
                            index=0,
                        )
                    )
                    - 1
                )
                if index >= 0:
                    scenario = st.session_state.scenarios[index]
                else:
                    scenario = st.session_state.last_scenario
                return index, scenario

            if len(st.session_state.scenarios) >= 2 and st.checkbox(
                "Compare scenarios?"
            ):
                product = st.selectbox(
                    "Pick the product 👇",
                    [""] + instance.selected_products,
                    key=f"scenario_compare_product",
                )
                location = st.selectbox(
                    "Pick the location 👇",
                    [""]
                    + instance.locations_with.get(product, instance.selected_locations),
                    key=f"scenario_compare_location",
                )
                product_location = (product, location)

                col1, col2 = st.columns(2)

                with col1:
                    _, scenario = select_scenario(f"Select scenario A 👇", "scenario_A")
                    ampl.reset()
                    ampl.eval(scenario["snapshot"])

                    reports = Reports(instance, ampl, key="scenario_A")
                    reports.material_balance_report(
                        include_demand=True, product_location=product_location
                    )

                with col2:
                    _, scenario = select_scenario(f"Select scenario B 👇", "scenario_B")
                    ampl.reset()
                    ampl.eval(scenario["snapshot"])

                    reports = Reports(instance, ampl, key="scenario_B")
                    reports.material_balance_report(
                        include_demand=True, product_location=product_location
                    )
            else:
                _, scenario = select_scenario()
                ampl.reset()
                ampl.eval(scenario["snapshot"])

                # Reports
                st.markdown("## Reports")
                reports = Reports(problem_number, instance, ampl)

                st.markdown("### Demand Report")
                if problem_number == 1:
                    reports.demand_report(by="location")
                else:
                    reports.demand_report(by="customer")

                st.markdown("### Material Balance Report")
                reports.material_balance_report()

                if include_homework3:
                    st.markdown("### Network Graph")
                    reports.network_report()

                if include_homework2:
                    st.markdown("### Resource Utilization Report")
                    reports.resource_utilization_report()

            if st.session_state.scenarios != [] and st.button("Delete all scenarios"):
                st.session_state.scenarios = []

    st.markdown(
        """##### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/supply_chain)] [[ChatGPT Solving Homework exercises]](https://chatgpt.com/share/e6f49ec8-3931-4586-b944-f104aebacd46)"""
    )
