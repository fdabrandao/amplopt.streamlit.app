from amplpy import AMPL
import streamlit as st
import pandas as pd
import random
import time
import json
import os
import io
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@st.cache_data()
def load_all_cities():
    # uscities.csv was obtained from https://simplemaps.com/data/us-cities
    all_cities_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "uscities.csv"))
    all_cities_df = all_cities_df[["state_name", "city", "lat", "lng"]]
    all_cities_df.columns = ["State", "City", "lat", "lon"]
    states = sorted(set(all_cities_df["State"]))
    return states, all_cities_df


@st.cache_data
def default_data(all_cities_df, state):
    cities_df = all_cities_df[all_cities_df["State"] == state].copy()
    cities_df.drop_duplicates(subset="City", keep="first", inplace=True)
    cities = list(cities_df["City"])
    facilities = random.sample(cities, 3)
    customers = random.sample(cities, 6)
    capacity = {f: 1500 + random.randint(0, 10) * 50 for f in cities}
    fixed_cost = {f: random.randint(2, 10) * 100000 for f in cities}
    min_demand = {c: 100 + random.randint(0, 10) * 25 for c in cities}
    max_demand = {c: min_demand[c] + 100 + random.randint(0, 10) * 25 for c in cities}
    return {
        "cities_df": cities_df,
        "facilities": facilities,
        "customers": customers,
        "capacity": capacity,
        "fixed_cost": fixed_cost,
        "min_demand": min_demand,
        "max_demand": max_demand,
    }


def haversine_distance(p1, p2):
    """
    Calculate the great-circle distance between two points
    on the Earth given their longitudes and latitudes in degrees.
    """
    from math import radians, sin, cos, sqrt, atan2

    (lat1, lon1), (lat2, lon2) = p1, p2
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in kilometers

    return distance


def main():
    # Streamlit app
    st.header("🏭 Stochastic Facility Location")

    st.write("## Facility and Customer Locations")

    states, all_cities_df = load_all_cities()

    state = st.selectbox(
        "Select the state 👇", states, key="state", index=states.index("Texas")
    )
    cities_df = default_data(all_cities_df, state)["cities_df"]
    default_facilities = default_data(all_cities_df, state)["facilities"]
    default_customers = default_data(all_cities_df, state)["customers"]

    # Select locations for facilities

    facility_locations = st.multiselect(
        "Select cities for facilities 👇",
        cities_df["City"],
        default=default_facilities,
        max_selections=10,
    )
    facility_locations = list(sorted(set(facility_locations)))
    if facility_locations == []:
        st.error("Please select at least one city.")
        st.stop()

    # Select locations for customers

    customer_locations = st.multiselect(
        "Select cities for customers 👇",
        cities_df["City"],
        default=default_customers,
        max_selections=10,
    )
    customer_locations = list(sorted(set(customer_locations)))
    if customer_locations == []:
        st.error("Please select at least one city.")
        st.stop()

    # Display all locations

    facilities_df = cities_df[cities_df["City"].isin(facility_locations)].copy()
    facilities_df["color"] = "#00FF00"
    facilities_df["size"] = 10000
    facilities_df["Type"] = "Facility"
    customers_df = cities_df[cities_df["City"].isin(customer_locations)].copy()
    customers_df["color"] = "#FF0000"
    customers_df["size"] = 1
    customers_df["Type"] = "Customer"
    locations = pd.concat([facilities_df, customers_df], axis=0)
    st.map(
        locations[["lat", "lon", "color", "size"]].copy(),
        latitude="lat",
        longitude="lon",
        color="color",
        size="size",
    )

    # Adjust the variable cost based on haversine distance

    st.write("### Variable cost:")

    distance_cost = st.slider(
        "Distance cost per kilometer 👇", min_value=1, max_value=100, step=1
    )

    coords = {row["City"]: (row["lat"], row["lon"]) for _, row in locations.iterrows()}
    variable_cost = pd.DataFrame(
        [
            {
                "Facility": facility,
                "Customer": customer,
                "Distance": distance_cost
                * haversine_distance(coords[facility], coords[customer]),
            }
            for facility in facility_locations
            for customer in customer_locations
        ]
    ).set_index(["Facility", "Customer"])

    # Display variable cost matrix

    st.write(
        variable_cost.reset_index().pivot_table(
            index="Facility", columns="Customer", values="Distance"
        )
    )

    # Set capacities and costs for the facilities

    st.write("## Facility capacities and fixed costs")

    df = facilities_df[["City"]].copy()
    capacity = default_data(all_cities_df, state)["capacity"]
    fixed_cost = default_data(all_cities_df, state)["fixed_cost"]
    df["Capacity"] = [capacity[f] for f in df["City"]]
    df["FixedCost"] = [fixed_cost[f] for f in df["City"]]
    df.set_index(["City"], inplace=True)
    edited_facilities_df = st.data_editor(df, disabled=["City"])

    # Adjust lower and upper bounds for customer demand

    st.write("## Customer demand")

    df = customers_df[["City"]].copy()
    min_demand = default_data(all_cities_df, state)["min_demand"]
    max_demand = default_data(all_cities_df, state)["max_demand"]
    df["MinDemand"] = [min_demand[c] for c in df["City"]]
    df["MaxDemand"] = [max_demand[c] for c in df["City"]]
    df.set_index(["City"], inplace=True)
    edited_customers_df = st.data_editor(df, disabled=["City"])

    # Choose the number of scenarios to consider

    st.write("## Scenarios")

    n_scenarios = st.slider(
        "Number of scenarios to generate 👇",
        min_value=5,
        max_value=25,
        step=1,
    )

    @st.cache_data
    def generate_scenarios(row, n):
        min_val = int(row["MinDemand"])
        max_val = int(row["MaxDemand"])
        scenarios = [
            {
                "City": row["City"],
                "Scenario": f"S{i+1}",
                "Demand": random.randint(min_val, max_val),
            }
            for i in range(n)
        ]
        return pd.DataFrame(scenarios)

    customer_demand_df = pd.concat(
        [
            generate_scenarios(row, n_scenarios)
            for _, row in edited_customers_df.reset_index().iterrows()
        ],
        axis=0,
    )
    customer_demand_df = customer_demand_df.pivot(
        index="City", columns="Scenario", values="Demand"
    )
    st.write(customer_demand_df)

    # Prepare data for serialization

    data = {
        "FACILITIES": list(edited_facilities_df.index),
        "CUSTOMERS": list(edited_customers_df.index),
        "SCENARIOS": [f"S{i+1}" for i in range(n_scenarios)],
        "prob": {f"S{i+1}": 1 / n_scenarios for i in range(n_scenarios)},
        "fixed_cost": edited_facilities_df[["FixedCost"]],
        "facility_capacity": edited_facilities_df[["Capacity"]],
        "variable_cost": variable_cost,
        "customer_demand": customer_demand_df,
    }

    # Serialize the instance data in JSON

    def serialize_input(data):
        return json.dumps(
            data,
            default=lambda x: (
                x.to_json(orient="table") if isinstance(x, pd.DataFrame) else str(x)
            ),
        )

    json_data = serialize_input(data)

    # Write json file for debugging
    # open("input.json", "w").write(json_data)

    # Pick the solver to use

    solvers = ["gurobi", "cplex", "highs"]
    solver = st.selectbox("Pick the solver to use 👇", solvers, key="solver")

    # Pick the location to solve the problems

    NEXTMV_APP_ID = "first-app"
    NEXTMV_API_KEY = os.environ.get("NEXTMV_API_KEY", "")
    if NEXTMV_API_KEY == "":
        NEXTMV_API_KEY = st.query_params.get("NEXTMV_API_KEY", "")
    if NEXTMV_API_KEY != "":
        worker_locations = ["nextmv", "nextmv-async", "locally"]
        worker_location = st.selectbox(
            "Pick where to run 👇", worker_locations, key="worker_location"
        )
    else:
        worker_location = "locally"

    # Pick approach

    approaches = ["stochastic", "individual scenarios"]
    approach = st.selectbox("Pick a solution approach 👇", approaches, key="approach")

    def nextmv_job_async(data: dict, solver: str) -> str:
        """
        Solve the problem asynchronously using the Nextmv Cloud API and return the run_id.
        """
        from nextmv.cloud import Application, Client

        client = Client(api_key=NEXTMV_API_KEY)
        app = Application(client=client, id=NEXTMV_APP_ID)
        run_id = app.new_run(
            input=serialize_input(data),
            options={"provider": solver},
        )
        return run_id

    def retrieve_nextmv_run_response(run_id: str) -> dict:
        """
        Retrieve the result from a run_id on Nextmv.
        """
        from nextmv.cloud import Application, Client, PollingOptions

        client = Client(api_key=NEXTMV_API_KEY)
        app = Application(client=client, id=NEXTMV_APP_ID)
        result = app.run_result_with_polling(
            run_id=run_id,
            polling_options=PollingOptions(),
        )
        return result.to_dict()

    def nextmv_job(data: dict, solver: str) -> dict:
        """
        Solve the problem using the Nextmv Cloud API and return the result.
        """
        run_id = nextmv_job_async(data, solver)
        return retrieve_nextmv_run_response(run_id)

    def nextmv_job_with_result(data: dict, solver: str) -> dict:
        """
        Solve the problem using the Nextmv Cloud API and return the result.
        """
        from nextmv.cloud import Application, Client, PollingOptions

        client = Client(api_key=NEXTMV_API_KEY)
        app = Application(client=client, id="first-app")
        result = app.new_run_with_result(
            input=serialize_input(data),
            polling_options=PollingOptions(),
            run_options={"provider": solver},
        )
        return result.to_dict()

    def solve_locally(data: dict, solver: str) -> tuple:
        t0 = time.time()
        ampl = AMPL()
        ampl.option["solver"] = solver
        ampl.cd(os.path.dirname(__file__))
        ampl.read("floc_bend.mod")
        ampl.set["FACILITIES"] = data["FACILITIES"]
        ampl.set["CUSTOMERS"] = data["CUSTOMERS"]
        ampl.set["SCENARIOS"] = data["SCENARIOS"]
        ampl.param["prob"] = data["prob"]
        ampl.param["fixed_cost"] = data["fixed_cost"]
        ampl.param["facility_capacity"] = data["facility_capacity"]
        ampl.param["variable_cost"] = data["variable_cost"]
        ampl.param["customer_demand"] = data["customer_demand"]
        output = ampl.get_output("include floc_bend.run;")
        run_duration = time.time() - t0
        solution = ampl.get_data("facility_open").to_pandas()
        total_cost = ampl.get_value("total_cost")
        return {
            "output": output,
            "run_duration": run_duration,
            "solution": solution,
            "total_cost": total_cost,
        }

    def extract_nextmv_solution(response):
        output = response["output"]["statistics"]["result"]["custom"]["solve_output"]
        run_duration = response["output"]["statistics"]["run"]["duration"]
        solution = pd.read_json(
            io.StringIO(response["output"]["solutions"][0]["facility_open"]),
            orient="table",
        )
        total_cost = response["output"]["solutions"][0]["total_cost"]
        return {
            "output": output,
            "run_duration": run_duration,
            "solution": solution,
            "total_cost": total_cost,
        }

    def solve(worker_location, solver, data):
        if worker_location == "locally":
            return solve_locally(data, solver)
        elif worker_location.startswith("nextmv"):
            response = nextmv_job(data, solver)
            return extract_nextmv_solution(response)
        else:
            st.error("Invalid worker location")
            st.stop()

    def solve_all(worker_location, solver, jobs):
        if worker_location == "nextmv-async":
            run_ids = {}
            for job, data in jobs.items():
                run_ids[job] = nextmv_job_async(data, solver)
            results = {}
            for job in run_ids:
                response = retrieve_nextmv_run_response(run_ids[job])
                results[job] = extract_nextmv_solution(response)
            return results

        results = {}
        for job, data in jobs.items():
            results[job] = solve(worker_location, solver, data)
        return results

    def display_solution(
        result,
        show_map=False,
        show_solve_output=False,
    ):
        # Display the solution
        st.write(f"Solver: {solver}")
        st.write(f"Run duration: {result['run_duration']:.2f}s")
        st.write(f"Total Cost: {result['total_cost']}")
        st.write("Facilities:")
        solution = result["solution"]
        st.write(solution)

        if show_map:
            sol_locations = locations[locations["City"].isin(solution.index)].copy()
            sol_locations.set_index(["City"], inplace=True)
            sol_locations["color"] = sol_locations.index.map(
                lambda c: (
                    "#0000FF" if solution.loc[c, "facility_open"] >= 0.5 else "#000000"
                )
            )
            st.map(
                sol_locations[["lat", "lon", "color", "size"]],
                latitude="lat",
                longitude="lon",
                color="color",
                size="size",
            )

        if show_solve_output:
            # Display the solve process output
            st.write("### Solve process output")
            st.write(f"```\n{result['output']}\n```")

    if approach == "stochastic":
        result = solve(worker_location, solver, data)
        st.write("## Solution")
        display_solution(result, show_map=True, show_solve_output=True)
    elif approach == "individual scenarios":
        jobs = {}
        for scenario in data["SCENARIOS"]:
            data_scenario = data.copy()
            data_scenario["SCENARIOS"] = [scenario]
            data_scenario["prob"] = {scenario: 1}
            data_scenario["customer_demand"] = data["customer_demand"][[scenario]]
            jobs[scenario] = data_scenario
        results = solve_all(worker_location, solver, jobs)
        solutions = {}
        for scenario in data["SCENARIOS"]:
            result = results[scenario]
            for index, row in result["solution"].iterrows():
                solutions[index, scenario] = row["facility_open"]
            st.write(f"## Solution for {scenario}")
            display_solution(result, show_map=False, show_solve_output=False)

        st.write("## Summary")
        df = pd.Series(solutions).reset_index()
        df.columns = ["City", "Scenario", "Open"]
        df = df.pivot_table(index="City", columns="Scenario", values="Open")
        df["Average"] = df.mean(axis=1)
        st.write(df)
    else:
        st.error("Invalid approach")
        st.stop()

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/facility_location)]"""
    )
