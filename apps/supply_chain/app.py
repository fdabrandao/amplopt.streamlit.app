import streamlit as st
from amplpy import AMPL
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import re
from ..common import solver_selector


class InputData:
    DEMAND_COLUMNS = ["Product", "Location", "Period", "Quantity", "DemandType"]
    STARTING_INVENTORY_COLUMNS = ["Product", "Location", "Quantity"]
    RATE_COLUMNS = ["Product", "Resource", "Rate", "Location", "Details"]
    AVAILABLE_CAPACITY_COLUMNS = ["Resource", "Location", "TotalCapacity", "Unit"]
    TRANSPORTATION_COSTS_COLUMNS = ["FromLocation", "ToLocation", "Allowed?", "Cost"]

    def __init__(self, xlsx_fname):
        self.dfs = pd.read_excel(
            xlsx_fname,
            sheet_name=None,
        )
        self.dfs["Demand"]["Period"] = pd.to_datetime(self.dfs["Demand"]["Period"])

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
        self.all_periods = list(sorted(set(self.demand["Period"])))
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

        # FIXME: this will be needed later
        # self.selected_resources = st.multiselect(
        #     "Resources:", self.all_resources, default=self.all_resources
        # )
        # # FIXME: Nothing to filter yet

        # cols = st.columns(len(self.all_locations))
        # resources_at = {}
        # for i, location in enumerate(self.selected_locations):
        #     with cols[i]:
        #         resources_at[location] = st.multiselect(
        #             f"Resources at {location}:",
        #             self.all_resources,
        #             default=self.all_resources_at.get(location, []),
        #         )
        # # Filter resources at each location
        # pairs = [
        #     (resource, location)
        #     for location in resources_at
        #     for resource in resources_at[location]
        # ]
        # self.rate = self.rate[
        #     self.rate["Resource"].isin(
        #         [f"{resource}_{location}" for (resource, location) in pairs]
        #     )
        # ]
        # mask = self.available_capacity.apply(
        #     lambda row: (row["Resource"], row["Location"]) in pairs, axis=1
        # )
        # self.available_capacity = self.available_capacity[mask]

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

        self.selected_suppliers = st.multiselect(
            "Suppliers:", self.all_suppliers, default=self.all_suppliers
        )
        # FIXME: Nothing to filter yet

        # Restrict table
        self.products_locations = list(
            sorted(
                set(zip(self.demand["Product"], self.demand["Location"]))
                | set(
                    zip(
                        self.starting_inventory["Product"],
                        self.starting_inventory["Location"],
                    )
                )
            )
        )
        self.products_at = defaultdict(lambda: [])
        self.locations_with = defaultdict(lambda: [])
        for product, location in self.products_locations:
            self.products_at[location].append(product)
            self.locations_with[product].append(location)

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

        # FIXME: this will be needed later
        # st.write("Rate:")
        # self.rate = data_editor(self.rate, ["Rate"])

        # st.write("AvailableCapacity:")
        # self.available_capacity = data_editor(
        #     self.available_capacity, ["TotalCapacity"]
        # )

        st.write("TransportationCosts:")
        self.transportation_costs = data_editor(self.transportation_costs, ["Cost"])


class Reports:
    def __init__(self, instance, ampl):
        self.instance = instance
        self.ampl = ampl

    def planning_view(
        self, key, df, view_func, all_products=False, all_locations=False
    ):
        if all_products:
            product = ""
        else:
            product = st.selectbox(
                "Pick the product 👇",
                [""] + self.instance.selected_products,
                key=f"{key}_view_product",
            )
        if all_locations:
            location = ""
        else:
            location = st.selectbox(
                "Pick the location 👇",
                [""]
                + self.instance.locations_with.get(
                    product, self.instance.selected_locations
                ),
                key=f"{key}_view_location",
            )
        label = ""
        filter = True
        if product != "":
            filter = df["Product"] == product
            label = product
        if location != "":
            filter = (df["Location"] == location) & filter
            if label == "":
                label = location
            else:
                label = f"{product} at {location}"
        if filter is True:
            view_func(df, label)
        else:
            view_func(df[filter], label)

    def demand_report(self):
        demand_df = self.ampl.get_data("Demand", "MetDemand", "UnmetDemand").to_pandas()
        demand_df.reset_index(inplace=True)
        demand_df.columns = ["Product", "Location", "Period"] + list(
            demand_df.columns[3:]
        )

        def demand_planning_view(df, label):
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

            df = pivot_table.T
            fig, ax = plt.subplots(figsize=(12, 3))
            # Stacking 'Met Demand' on top of 'Demand' and 'Unmet Demand' on top of 'Met Demand'
            ax.bar(
                df.columns,
                df.loc["Demand", :],
                label="Demand",
                edgecolor="black",
                linewidth=1.5,
                facecolor="none",
            )
            metdemand_bars = ax.bar(
                df.columns,
                df.loc["MetDemand", :],
                label="MetDemand",
                color="green",
            )
            unmetdemand_bars = ax.bar(
                df.columns,
                df.loc["UnmetDemand", :],
                bottom=df.loc["MetDemand", :],
                label="UnmetDemand",
                color="red",
            )

            # Adding labels and title
            ax.set_ylabel("Units")
            ax.set_title(f"{label} Demand Overview")
            ax.legend()

            # Adding text inside the bars for 'Met Demand' and 'Unmet Demand'
            for bar in metdemand_bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{yval}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

            for bar in unmetdemand_bars:
                yval = bar.get_height()
                if yval > 0:  # Only display if there's a noticeable unmet demand
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_y() - yval / 2,
                        f"{yval}",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )

            # Show the plot
            st.pyplot(plt)
            # Show the table
            st.dataframe(pivot_table.T)

        view = st.selectbox(
            "Demand Report",
            [
                "Planning View",
                "Planning View Per Product",
                "Planning View Per Location",
                "Full Report",
            ],
        )

        if view == "Planning View":
            self.planning_view("demand", demand_df, demand_planning_view)
        elif view == "Planning View Per Product":
            self.planning_view(
                "demand", demand_df, demand_planning_view, all_locations=True
            )
        elif view == "Planning View Per Location":
            self.planning_view(
                "demand", demand_df, demand_planning_view, all_products=True
            )
        else:
            st.dataframe(demand_df, hide_index=True)

    def material_balance_report(self):
        material_df = self.ampl.get_data(
            "StartingInventory", "MetDemand", "Production", "EndingInventory"
        ).to_pandas()
        material_df.reset_index(inplace=True)
        material_df.columns = ["Product", "Location", "Period"] + list(
            material_df.columns[3:]
        )

        view = st.selectbox(
            "Material Balance Report",
            [
                "Planning View",
                "Planning View Per Product",
                "Planning View Per Location",
                "Full Report",
            ],
        )

        def material_balance(df, label):
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

            df = pivot_table.T
            fig, ax = plt.subplots(figsize=(12, 3))

            # Plot lines for Starting Inventory, Production, and Ending Inventory
            ax.plot(
                df.columns,
                df.loc["StartingInventory", :],
                label="StartingInventory",
                marker="o",
            )
            ax.plot(df.columns, df.loc["Production", :], label="Production", marker="o")
            ax.plot(
                df.columns,
                df.loc["EndingInventory", :],
                label="EndingInventory",
                marker="o",
            )

            # Adding labels and title
            ax.set_ylabel("Units")
            ax.set_title(f"{label} Material Balance Overview")
            ax.legend()

            # Show the plot
            st.pyplot(plt)
            # Show the table
            st.dataframe(pivot_table.T)

        if view == "Planning View":
            self.planning_view("material", material_df, material_balance)
        elif view == "Planning View Per Product":
            self.planning_view(
                "material", material_df, material_balance, all_locations=True
            )
        elif view == "Planning View Per Location":
            self.planning_view(
                "material", material_df, material_balance, all_products=True
            )
        else:
            st.dataframe(material_df, hide_index=True)


def main():
    st.title("📦 Supply Chain Optimization")

    instance = InputData(
        os.path.join(os.path.dirname(__file__), "InputDataProductionSolver.xlsx")
    )

    with st.expander("Dimensions"):
        instance.filter_dimensions()

    with st.expander("Data"):
        instance.edit_data()

    model = r"""
        set PRODUCTS;  # Set of products
        set LOCATIONS;  # Set of distribution or production locations
        set PRODUCTS_LOCATIONS within {PRODUCTS, LOCATIONS};  # Restrict table
        set PERIODS ordered;  # Ordered set of time periods for planning

        param Demand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0 default 0;
                # Demand for each product at each location during each time period
        var UnmetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Quantity of demand that is not met for a product at a location in a time period
        var MetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Quantity of demand that is met for a product at a location in a time period

        param InitialInventory{p in PRODUCTS, l in LOCATIONS} >= 0 default 0;
                # Initial inventory levels for each product at each location
        var StartingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Inventory at the beginning of each time period
        var EndingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Inventory at the end of each time period
        var Production{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Production volume for each product at each location during each time period

        param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
        param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)

        minimize TotalCost:
            sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS}
                (UnmetDemandPenalty * UnmetDemand[p, l, t] + EndingInventoryPenalty * EndingInventory[p, l, t]);
                # Objective function to minimize total costs associated with unmet demand and leftover inventory
    
        # s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
        # ... Exercise 1: Ensure that all demand is accounted for either as met or unmet.

        # s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
        # ... Exercise 2: Define how inventory is carried over from one period to the next.

        # s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
        # ... Exercise 3: Balance starting inventory and production against demand to determine ending inventory.
    """

    demand_fulfillment = r"""
        s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            Demand[p, l, t] = MetDemand[p, l, t] + UnmetDemand[p, l, t];
                # Ensure that all demand is accounted for either as met or unmet.
    """

    inventory_balance = r"""         
        s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            StartingInventory[p, l, t] =
                if ord(t) > 1 then
                    EndingInventory[p, l, prev(t)]
                else
                    InitialInventory[p, l];
                # Define how inventory is carried over from one period to the next.
    """

    stock_balance = r"""       
        s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            StartingInventory[p, l, t] + Production[p, l, t] - MetDemand[p, l, t] = EndingInventory[p, l, t];
                # Balance starting inventory and production against demand to determine ending inventory.
    """

    st.markdown("## Production Optimization")

    use_restrict_table = st.checkbox("Use restrict table Product x Locations")
    if use_restrict_table:

        def apply_restrict_table(m):
            m = m.replace(
                "p in PRODUCTS, l in LOCATIONS", "(p, l) in PRODUCTS_LOCATIONS"
            )
            return m

        model = apply_restrict_table(model)
        demand_fulfillment = apply_restrict_table(demand_fulfillment)
        inventory_balance = apply_restrict_table(inventory_balance)
        stock_balance = apply_restrict_table(stock_balance)

    st.code(model)

    demand = instance.demand[["Product", "Location", "Period", "Quantity"]].copy()
    starting_inventory = instance.starting_inventory[
        ["Product", "Location", "Quantity"]
    ].copy()
    demand["Period"] = demand["Period"].dt.strftime("%Y-%m-%d")
    periods = list(sorted(set(demand["Period"])))
    demand.set_index(["Product", "Location", "Period"], inplace=True)
    starting_inventory.set_index(["Product", "Location"], inplace=True)

    ampl = AMPL()
    ampl.eval(model)
    ampl.set["PRODUCTS"] = instance.selected_products
    ampl.set["LOCATIONS"] = instance.selected_locations
    ampl.set["PRODUCTS_LOCATIONS"] = instance.products_locations
    ampl.set["PERIODS"] = periods
    ampl.param["Demand"] = demand["Quantity"]
    ampl.param["InitialInventory"] = starting_inventory["Quantity"]

    def exercise(name, constraint, needs, help=""):
        if st.checkbox(f"Skip exercise", key=f"Skip {name}", value=True):
            ampl.eval(constraint)
        else:
            constraint = constraint[constraint.find("s.t.") :]
            constraint = constraint[: constraint.find("\n")] + "\n\t"
            answer = st.text_input(f"Implement the {name} below").strip()
            if answer != "" and not answer.endswith(";"):
                answer += "\n;"

            if answer != "":
                st.code(constraint + answer)
            else:
                st.code(constraint + "\t... the equation above goes here ...;")
            forbidden = ["model", "data", "include", "shell", "cd"]
            validation_report = ""

            answer_nospace = answer.replace(" ", "")
            incomplete = False
            for s in needs:
                passed = s.replace(" ", "") in answer_nospace
                if not passed:
                    incomplete = True
                validation_report += f"- {'✅' if passed else '❌'} uses `{s}`\n"
            st.markdown(validation_report)
            if help != "":
                st.info(help)

            if answer_nospace == "":
                st.error(f"Please write the equation above.")
            elif incomplete or any(s in answer_nospace for s in forbidden):
                st.error(f"Please complete the equation above.")
            else:
                output = ampl.get_output(constraint + answer + ";")
                if output != "":
                    output = re.sub(
                        r"\bfile\s*-\s*line\s+\d+\s+offset\s+\d+\b", "", output
                    ).strip()
                    st.error(f"❌ Syntax Error: {output}")
                else:
                    st.success(
                        "Great! No syntax errors! Check the results below to confirm if it is correct!"
                    )
                output = ampl.get_output("write 0;")
                if output != "" and not output.startswith("No files written"):
                    if "Error executing " in output:
                        output = output[output.find(":") + 1 :].strip()
                    st.error(f"❌ Error: {output}")

    st.markdown(
        """
        ### Exercise #1: Demand Balance Constraint
        
        🧑‍🏫 Ensure that all demand is accounted for either as met or unmet.
        """
    )
    exercise(
        "Demand Balance Constraint",
        demand_fulfillment,
        ["Demand[p, l, t]", "MetDemand[p, l, t]", "UnmetDemand[p, l, t]", "="],
    )

    st.markdown(
        """
        ### Exercise #2: Inventory Carryover Constraint
        
        🧑‍🏫 Define how inventory is carried over from one period to the next.
        """
    )
    exercise(
        "Inventory Carryover Constraint",
        inventory_balance,
        [
            "StartingInventory[p, l, t]",
            "EndingInventory[p, l, prev(t)]",
            "InitialInventory[p, l]",
            "if",
            "ord(t)",
            "then",
            "=",
        ],
        help="""
        The set `PERIODS` is an ordered set (declared as `set PERIODS ordered;`).
        This allows checking the order of a set element `t` with `ord(t)` (starting at 1),
        and access the previous and following elements with `prev(t)` and `next(t)`, respectively.
        Learn more about this in Chapter 5 of [The AMPL Book](https://ampl.com/ampl-book/).

        You will also be using an `if-then-else` statement. Its syntax as follows:
        `if <condition> then <value or expression> else <value or expression>`.
        Learn more about this in Chapter 7 of [The AMPL Book](https://ampl.com/ampl-book/).

        Note that the [The AMPL Book](https://ampl.com/ampl-book/) is a good reference to learn
        AMPL syntax but it is not up to date in terms of how AMPL should be used in production.
        For more modern usage examples see https://ampl.com/mo-book/ and https://ampl.com/colab/ where
        AMPL is used integrated with Python just like in this Streamlit app.
        """,
    )

    st.markdown(
        """
        ### Exercise #3: Material Balance Constraint
        
        🧑‍🏫 Balance starting inventory and production against demand to determine ending inventory.
        """
    )
    exercise(
        "Material Balance Constraint",
        stock_balance,
        [
            "StartingInventory[p, l, t]",
            "Production[p, l, t]",
            "MetDemand[p, l, t]",
            "EndingInventory[p, l, t]",
            "=",
        ],
    )

    st.markdown("## Solve")

    with st.expander("Adjust objective penalties"):
        col1, col2 = st.columns(2)
        with col1:
            ampl.param["UnmetDemandPenalty"] = st.slider(
                "UnmetDemandPenalty:",
                min_value=0,
                max_value=50,
                value=10,
            )

        with col2:
            ampl.param["EndingInventoryPenalty"] = st.slider(
                "EndingInventoryPenalty:",
                min_value=0,
                max_value=50,
                value=5,
            )

    # Select the solver to use
    solver, _ = solver_selector(mp_only=True)

    # Solve the problem
    output = ampl.solve(solver=solver, mp_options="outlev=1", return_output=True)
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

    # Reports
    st.markdown("## Reports")
    reports = Reports(instance, ampl)

    st.markdown("### Demand Report")
    reports.demand_report()

    st.markdown("### Material Balance Report")
    reports.material_balance_report()
