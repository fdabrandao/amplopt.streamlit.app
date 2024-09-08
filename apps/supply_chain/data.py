import streamlit as st
import pandas as pd
from collections import defaultdict


def data_editor(df, columns):
    return st.data_editor(
        df,
        disabled=[c for c in df.columns if c not in columns],
        hide_index=True,
    )


class InputData:
    DEMAND_COLUMNS = ["Product", "Location", "Period", "Quantity", "DemandType"]
    STARTING_INVENTORY_COLUMNS = ["Product", "Location", "Quantity"]
    PRODUCTION_RATE_COLUMNS = ["Product", "Resource", "Location", "Rate"]
    AVAILABLE_CAPACITY_COLUMNS = [
        "Resource",
        "Location",
        "TotalCapacityPerPeriod",
    ]
    TRANSPORTATION_COSTS_COLUMNS = ["FromLocation", "ToLocation", "Allowed?", "Cost"]
    TRANSFER_LANES_COLUMNS = ["Product", "FromLocation", "ToLocation"]
    TARGET_STOCK_COLUMNS = ["Product", "Location", "TargetStock"]
    LOCATION_CAPACITY_COLUMNS = ["Location", "MaxCapacity"]

    def __init__(self, xlsx_fname, class_number):
        self.class_number = class_number
        self.dfs = pd.read_excel(
            xlsx_fname,
            sheet_name=None,
        )
        self.dfs["Demand"]["Period"] = pd.to_datetime(self.dfs["Demand"]["Period"])

        def load_sheet(name, columns):
            if set(columns) - set(self.dfs[name].columns) != set():
                st.error(f"{name} sheet needs columns: {columns}")
                st.stop()
            return self.dfs[name][columns].dropna().copy()

        # Data
        self.demand = load_sheet("Demand", self.DEMAND_COLUMNS)
        self.starting_inventory = load_sheet(
            "StartingInventory", self.STARTING_INVENTORY_COLUMNS
        )
        self.production_rate = load_sheet("Rate", self.PRODUCTION_RATE_COLUMNS)
        self.production_rate["Resource"] = self.production_rate["Resource"].apply(
            lambda r: r.split("_")[0]
        )
        self.available_capacity = load_sheet(
            "AvailableCapacity", self.AVAILABLE_CAPACITY_COLUMNS
        )
        self.transportation_costs = load_sheet(
            "TransportationCosts", self.TRANSPORTATION_COSTS_COLUMNS
        )
        self.transfer_lanes = load_sheet("TransferLanes", self.TRANSFER_LANES_COLUMNS)
        self.target_stocks = load_sheet("TargetStocks", self.TARGET_STOCK_COLUMNS)
        self.location_capacity = load_sheet(
            "LocationCapacity", self.LOCATION_CAPACITY_COLUMNS
        )

        # Dimensions
        self.all_products = list(sorted(set(self.demand["Product"])))
        self.all_components = ["Flour", "Sugar", "Chocolate"]
        self.all_locations = list(sorted(set(self.demand["Location"])))
        self.all_customers = ["Supermarket", "Restaurant", "Bulk"]
        self.all_resources = list(set(self.production_rate["Resource"]))
        self.all_resources_at = {l: set() for l in self.all_locations}
        for resource, location in zip(
            self.production_rate["Resource"], self.production_rate["Location"]
        ):
            self.all_resources_at[location].add(resource)
        for location in self.all_resources_at:
            self.all_resources_at[location] = list(
                sorted(self.all_resources_at[location])
            )
        self.all_periods = list(sorted(set(self.demand["Period"])))
        self.all_suppliers = ["Flour Shop", "Chocolate Shop"]

    def filter_dimensions(self):
        self._filter_dimensions_class1()
        if self.class_number <= 1:
            return

        self._filter_dimensions_class2()
        if self.class_number <= 2:
            return

    def edit_data(self):
        self._edit_data_class1()
        if self.class_number <= 1:
            return

        self._edit_data_class2()
        if self.class_number <= 2:
            return

        self._edit_data_class3()
        if self.class_number <= 3:
            return

    def _filter_dimensions_class1(self):
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
            self.production_rate = self.production_rate[
                self.production_rate["Product"].isin(self.selected_products)
            ]
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
            self.production_rate = self.production_rate[
                self.production_rate["Location"].isin(self.selected_locations)
            ]
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
                | set(
                    zip(
                        self.production_rate["Product"],
                        self.production_rate["Location"],
                    )
                )
            )
        )
        self.products_at = defaultdict(lambda: [])
        self.locations_with = defaultdict(lambda: [])
        for product, location in self.products_locations:
            self.products_at[location].append(product)
            self.locations_with[product].append(location)

    def _edit_data_class1(self):
        st.write("Demand:")
        self.demand = data_editor(self.demand, ["Quantity"])

        st.write("InitialInventory:")
        self.starting_inventory = data_editor(self.starting_inventory, ["Quantity"])

    def _filter_dimensions_class2(self):
        self.selected_resources = st.multiselect(
            "Resources:", self.all_resources, default=self.all_resources
        )

        cols = st.columns(len(self.all_locations))
        self.resources_at = {}
        for i, location in enumerate(self.selected_locations):
            with cols[i]:
                self.resources_at[location] = st.multiselect(
                    f"Resources at {location}:",
                    self.all_resources,
                    default=self.all_resources_at.get(location, []),
                )
        # Filter resources at each location
        self.resource_location_pairs = [
            (resource, location)
            for location in self.resources_at
            for resource in self.resources_at[location]
        ]
        self.production_rate = self.production_rate[
            self.production_rate.apply(
                lambda row: (row["Resource"], row["Location"])
                in self.resource_location_pairs,
                axis=1,
            )
        ]

        mask = self.available_capacity.apply(
            lambda row: (row["Resource"], row["Location"])
            in self.resource_location_pairs,
            axis=1,
        )
        self.available_capacity = self.available_capacity[mask]

        mask = self.transfer_lanes.apply(
            lambda row: row["Product"] in self.selected_products
            and row["FromLocation"] in self.selected_locations
            and row["ToLocation"] in self.selected_locations,
            axis=1,
        )
        self.transfer_lanes = self.transfer_lanes[mask]

    def _edit_data_class2(self):
        st.write("ProductionRate:")
        self.production_rate = data_editor(self.production_rate, ["Rate"])

        st.write("AvailableCapacity:")
        self.available_capacity = data_editor(
            self.available_capacity, ["TotalCapacityPerPeriod"]
        )

        st.write("TransferLanes:")
        self.transfer_lanes = st.data_editor(
            self.transfer_lanes,
            hide_index=True,
            column_config={
                "Product": st.column_config.SelectboxColumn(
                    options=self.selected_products,
                    default="",
                ),
                "FromLocation": st.column_config.SelectboxColumn(
                    options=self.selected_locations,
                    default="",
                ),
                "ToLocation": st.column_config.SelectboxColumn(
                    options=self.selected_locations,
                    default="",
                ),
            },
        )

        st.write("TargetStock:")
        self.target_stocks = data_editor(self.target_stocks, ["TargetStock"])

    def _filter_dimensions_class3(self):
        pass

    def _edit_data_class3(self):
        st.write("TransportationCosts:")
        self.transportation_costs = data_editor(self.transportation_costs, ["Cost"])
