import streamlit as st
from amplpy import AMPL
import pandas as pd
import os


class InputData:
    DEMAND_COLUMNS = ["Product", "Location", "Period", "Quantity", "DemandType"]
    STARTING_INVENTORY_COLUMNS = ["Product", "Location", "Period", "Quantity"]
    RATE_COLUMNS = ["Product", "Resource", "Rate", "Location", "Details"]
    AVAILABLE_CAPACITY_COLUMNS = ["Resource", "Location", "TotalCapacity", "Unit"]
    TRANSPORTATION_COSTS_COLUMNS = ["FromLocation", "ToLocation", "Allowed?", "Cost"]

    def __init__(self, xlsx_fname):
        self.dfs = pd.read_excel(
            xlsx_fname,
            sheet_name=None,
        )
        self.dfs["Demand"]["Period"] = pd.to_datetime(self.dfs["Demand"]["Period"])
        self.dfs["StartingInventory"]["Period"] = pd.to_datetime(
            self.dfs["StartingInventory"]["Period"]
        )

        def load_sheet(name, columns):
            if set(columns) - set(self.dfs[name].columns) != set():
                st.error(f"{name} sheet needs columns: {columns}")
                st.stop()
            return self.dfs[name][columns].copy()

        # Data
        self.demand = load_sheet("Demand", self.DEMAND_COLUMNS)
        self.starting_inventory = load_sheet(
            "StartingInventory", self.STARTING_INVENTORY_COLUMNS
        )
        self.rate = load_sheet("Rate", self.RATE_COLUMNS)
        self.available_capacity = load_sheet(
            "AvailableCapacity", self.AVAILABLE_CAPACITY_COLUMNS
        )
        self.transportation_costs = load_sheet(
            "TransportationCosts", self.TRANSPORTATION_COSTS_COLUMNS
        )

        # Dimensions
        self.all_products = list(sorted(set(self.demand["Product"])))
        self.all_components = ["Flour", "Sugar", "Chocolate"]
        self.all_locations = list(sorted(set(self.demand["Location"])))
        self.all_customers = ["Supermarket", "Restaurant", "Bulk"]
        self.all_resources = list(
            sorted(set([pair.split("_")[0] for pair in self.rate["Resource"]]))
        )
        self.all_resources_at = {l: set() for l in self.all_locations}
        for pair in self.rate["Resource"]:
            resource, location = pair.split("_")
            self.all_resources_at[location].add(resource)
        for location in self.all_resources_at:
            self.all_resources_at[location] = list(
                sorted(self.all_resources_at[location])
            )
        self.all_periods = list(
            sorted(set(self.demand["Period"]) | set(self.starting_inventory["Period"]))
        )
        self.all_suppliers = ["Flour Shop", "Chocolate Shop"]

    def filter_dimensions(self):
        cols = st.columns(3)
        with cols[0]:
            self.selected_products = st.multiselect(
                "Products:", self.all_products, default=self.all_products
            )
            # Filter products
            self.demand = self.demand[
                self.demand["Product"].isin(self.selected_products)
            ]
            self.starting_inventory = self.starting_inventory[
                self.starting_inventory["Product"].isin(self.selected_products)
            ]
            self.rate = self.rate[self.rate["Product"].isin(self.selected_products)]
        with cols[1]:
            self.selected_components = st.multiselect(
                "Components:", self.all_components, default=self.all_components
            )
            # FIXME: Nothing to filter yet
        with cols[2]:
            self.selected_locations = st.multiselect(
                "Locations:", self.all_locations, default=self.all_locations
            )
            # Filter locations
            self.demand = self.demand[
                self.demand["Location"].isin(self.selected_locations)
            ]
            self.starting_inventory = self.starting_inventory[
                self.starting_inventory["Location"].isin(self.selected_locations)
            ]
            self.rate = self.rate[self.rate["Location"].isin(self.selected_locations)]
            self.available_capacity = self.available_capacity[
                self.available_capacity["Location"].isin(self.selected_locations)
            ]
            self.transportation_costs = self.transportation_costs[
                self.transportation_costs["ToLocation"].isin(self.selected_locations)
            ]

        self.selected_customers = st.multiselect(
            "Customers:", self.all_customers, default=self.all_customers
        )
        # FIXME: Nothing to filter yet

        self.selected_resources = st.multiselect(
            "Resources:", self.all_resources, default=self.all_resources
        )
        # FIXME: Nothing to filter yet

        cols = st.columns(len(self.all_locations))
        resources_at = {}
        for i, location in enumerate(self.selected_locations):
            with cols[i]:
                resources_at[location] = st.multiselect(
                    f"Resources at {location}:",
                    self.all_resources,
                    default=self.all_resources_at.get(location, []),
                )
        # Filter resources at each location
        pairs = [
            (resource, location)
            for location in resources_at
            for resource in resources_at[location]
        ]
        self.rate = self.rate[
            self.rate["Resource"].isin(
                [f"{resource}_{location}" for (resource, location) in pairs]
            )
        ]
        mask = self.available_capacity.apply(
            lambda row: (row["Resource"], row["Location"]) in pairs, axis=1
        )
        self.available_capacity = self.available_capacity[mask]

        date_range = (
            min(self.all_periods).to_pydatetime(),
            max(self.all_periods).to_pydatetime(),
        )
        self.selected_range = st.slider(
            "Periods:",
            min_value=date_range[0],
            max_value=date_range[1],
            value=(date_range[0], date_range[1]),
            format="YYYY-MM-DD",
        )
        # Filter periods
        self.demand = self.demand[
            (self.demand["Period"] >= self.selected_range[0])
            & (self.demand["Period"] <= self.selected_range[1])
        ]
        self.starting_inventory = self.starting_inventory[
            (self.starting_inventory["Period"] >= self.selected_range[0])
            & (self.starting_inventory["Period"] <= self.selected_range[1])
        ]

        self.selected_suppliers = st.multiselect(
            "Suppliers:", self.all_suppliers, default=self.all_suppliers
        )
        # FIXME: Nothing to filter yet

    def edit_data(self):
        def data_editor(df, columns):
            return st.data_editor(
                df,
                disabled=[c for c in df.columns if c not in columns],
                hide_index=True,
            )

        st.write("Demand:")
        self.demand = data_editor(self.demand, ["Quantity"])

        st.write("StartingInventory:")
        self.starting_inventory = data_editor(self.starting_inventory, ["Quantity"])

        st.write("Rate:")
        self.rate = data_editor(self.rate, ["Rate"])

        st.write("AvailableCapacity:")
        self.available_capacity = data_editor(
            self.available_capacity, ["TotalCapacity"]
        )

        st.write("TransportationCosts:")
        self.transportation_costs = data_editor(self.transportation_costs, ["Cost"])


def main():
    st.markdown(
        """
    # 📦 Supply Chain Optimization
    
    
    """
    )

    instance = InputData(
        os.path.join(os.path.dirname(__file__), "InputDataProductionSolver.xlsx")
    )

    with st.expander("Dimensions"):
        instance.filter_dimensions()

    instance.edit_data()

    ampl = AMPL()
    ampl.eval(
        r"""
        set ProductLocationPeriod dimen 3;

        param Demand{ProductLocationPeriod};
        
        var UnmetDemand{ProductLocationPeriod} >= 0;
        var MetDemand{ProductLocationPeriod} >= 0;

        var StartingInventory{ProductLocationPeriod} >= 0;
        var EndingInventory{ProductLocationPeriod} >= 0;
        
        var Production{ProductLocationPeriod} >= 0;

        minimize Objective:
            sum {(p, l, t) in ProductLocationPeriod}
                (10 * UnmetDemand[p, l, t] + 5 * EndingInventory[p, l, t]);

        s.t. DemandBalance{(p, l, t) in ProductLocationPeriod}:
            Demand[p, l, t] = MetDemand[p, l, t] + UnmetDemand[p, l, t];
        """
    )
    st.write(instance.demand)
    st.write(
        instance.demand[["Product", "Location", "Period", "Quantity"]].set_index(
            ["Product", "Location", "Period"]
        )
    )
    df = instance.demand[["Product", "Location", "Period", "Quantity"]].copy()
    df["Period"] = df["Period"].dt.strftime("%Y-%m-%d")
    df.set_index(["Product", "Location", "Period"], inplace=True)
    ampl.set["ProductLocationPeriod"] = df.index
    ampl.param["Demand"] = df

    solvers = ["gurobi", "xpress", "cplex", "mosek", "copt", "highs", "scip", "cbc"]
    solver = st.selectbox("Pick the solver to use 👇", solvers, key="solver")
    if solver == "cplex":
        solver = "cplexmp"
    output = ampl.solve(solver=solver, mp_options="outlev=1", return_output=True)
    st.write(f"```\n{output}\n```")

    # Demand report
    df = ampl.get_data("Demand", "MetDemand", "UnmetDemand").to_pandas()
    df.reset_index(inplace=True)
    df.columns = ["Product", "Location", "Period"] + list(df.columns[3:])

    def demand_report(df):
        columns = [
            "Demand",
            "MetDemand",
            "UnmetDemand",
        ]
        pivot_table = pd.pivot_table(
            df,
            index="Period",  # Use 'Period' as the index
            values=columns,  # Specify the columns to aggregate
            aggfunc="sum",  # Use sum as the aggregation function
        )[columns]
        st.dataframe(pivot_table.T)

    view = st.selectbox(
        "Demand Report",
        [
            "Pivot Table",
            "Pivot Table Per Product",
            "Pivot Table Per Location",
            "Full Table",
        ],
    )

    if view == "Pivot Table":
        demand_report(df)
    elif view == "Pivot Table Per Product":
        for product in instance.selected_products:
            st.markdown(f"Product: {product}")
            demand_report(df[df["Product"] == product])
    elif view == "Pivot Table Per Location":
        for location in instance.selected_locations:
            st.markdown(f"Location: {location}")
            demand_report(df[df["Location"] == location])
    else:
        st.dataframe(df, hide_index=True)

    # Material balance report
    df = ampl.get_data(
        "StartingInventory", "MetDemand", "Production", "EndingInventory"
    ).to_pandas()
    df.reset_index(inplace=True)
    df.columns = ["Product", "Location", "Period"] + list(df.columns[3:])

    view = st.selectbox(
        "Material Balance Report",
        [
            "Pivot Table",
            "Pivot Table Per Product",
            "Pivot Table Per Location",
            "Full Table",
        ],
    )

    def material_balance(df):
        columns = [
            "StartingInventory",
            "MetDemand",
            "Production",
            "EndingInventory",
        ]
        pivot_table = pd.pivot_table(
            df,
            index="Period",  # Use 'Period' as the index
            values=columns,  # Specify the columns to aggregate
            aggfunc="sum",  # Use sum as the aggregation function
        )[columns]
        st.dataframe(pivot_table.T)

    if view == "Pivot Table":
        material_balance(df)
    elif view == "Pivot Table Per Product":
        for product in instance.selected_products:
            st.markdown(f"Product: {product}")
            material_balance(df[df["Product"] == product])
    elif view == "Pivot Table Per Location":
        for location in instance.selected_locations:
            st.markdown(f"Location: {location}")
            material_balance(df[df["Location"] == location])
    else:
        st.dataframe(df, hide_index=True)
