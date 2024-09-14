import streamlit as st
import pandas as pd
from collections import defaultdict


@st.cache_data
def read_excel(xlsx_fname):
    return pd.read_excel(
        xlsx_fname,
        sheet_name=None,
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

    def __init__(self, xlsx_fname, class_number, on_change=None):
        self.on_change = on_change
        self.class_number = class_number
        self.dfs = read_excel(xlsx_fname)
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

    def _data_editor(self, df, columns):
        return st.data_editor(
            df,
            disabled=[c for c in df.columns if c not in columns],
            hide_index=True,
            on_change=self.on_change,
        )

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

    @property
    def products_locations(self):
        return self._products_locations

    @products_locations.setter
    def products_locations(self, value):
        if not isinstance(value, list):
            raise ValueError("products_locations must be a list")
        self._products_locations = value
        self.products_at = defaultdict(lambda: [])
        self.locations_with = defaultdict(lambda: [])
        for product, location in self._products_locations:
            self.products_at[location].append(product)
            self.locations_with[product].append(location)

    def _filter_dimensions_class1(self):
        cols = st.columns(3)
        with cols[0]:
            self.selected_products = st.multiselect(
                "Products:",
                self.all_products,
                default=self.all_products,
                on_change=self.on_change,
            )
        with cols[1]:
            self.selected_components = st.multiselect(
                "Components:",
                self.all_components,
                default=self.all_components,
                on_change=self.on_change,
            )
            # FIXME: Nothing to filter yet
        with cols[2]:
            self.selected_locations = st.multiselect(
                "Locations:",
                self.all_locations,
                default=self.all_locations,
                on_change=self.on_change,
            )

        mask = self.demand.apply(
            lambda row: row["Product"] in self.selected_products
            and row["Location"] in self.selected_locations,
            axis=1,
        )
        self.demand = self.demand[mask]

        mask = self.starting_inventory.apply(
            lambda row: row["Product"] in self.selected_products
            and row["Location"] in self.selected_locations,
            axis=1,
        )
        self.starting_inventory = self.starting_inventory[mask]

        self.selected_customers = st.multiselect(
            "Customers:",
            self.all_customers,
            default=self.all_customers,
            on_change=self.on_change,
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
            on_change=self.on_change,
        )
        # Filter periods
        self.demand = self.demand[
            (self.demand["Period"] >= self.selected_range[0])
            & (self.demand["Period"] <= self.selected_range[1])
        ]

        self.selected_suppliers = st.multiselect(
            "Suppliers:",
            self.all_suppliers,
            default=self.all_suppliers,
            on_change=self.on_change,
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

    def _edit_data_class1(self):
        st.write("Demand:")
        self.demand = self._data_editor(self.demand, ["Quantity"])

        st.write("InitialInventory:")
        self.starting_inventory = self._data_editor(
            self.starting_inventory, ["Quantity"]
        )

    def _filter_dimensions_class2(self):
        self.selected_resources = st.multiselect(
            "Resources:",
            self.all_resources,
            default=self.all_resources,
            on_change=self.on_change,
        )

        cols = st.columns(len(self.all_locations))
        self.resources_at = {}
        for i, location in enumerate(self.selected_locations):
            with cols[i]:
                self.resources_at[location] = st.multiselect(
                    f"Resources at {location}:",
                    self.all_resources,
                    default=self.all_resources_at.get(location, []),
                    on_change=self.on_change,
                )
        # Filter resources at each location
        self.resource_location_pairs = [
            (resource, location)
            for location in self.resources_at
            for resource in self.resources_at[location]
        ]

        mask = self.production_rate.apply(
            lambda row: (row["Resource"], row["Location"])
            in self.resource_location_pairs
            and row["Product"] in self.selected_products,
            axis=1,
        )
        self.production_rate = self.production_rate[mask]

        # Expand products_locations
        self.products_locations = list(
            set(self.products_locations)
            | set(
                zip(
                    self.production_rate["Product"],
                    self.production_rate["Location"],
                )
            )
        )

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

        mask = self.target_stocks.apply(
            lambda row: row["Product"] in self.selected_products
            and row["Location"] in self.selected_locations,
            axis=1,
        )
        self.target_stocks = self.target_stocks[mask]

        self.location_capacity = self.location_capacity[
            self.location_capacity["Location"].isin(self.selected_locations)
        ]

    def _edit_data_class2(self):
        st.write("ProductionRate:")
        self.production_rate = self._data_editor(self.production_rate, ["Rate"])

        st.write("AvailableCapacity:")
        self.available_capacity = self._data_editor(
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
            on_change=self.on_change,
        )

        st.write("TargetStock:")
        self.target_stocks = self._data_editor(self.target_stocks, ["TargetStock"])

        st.write("MaxCapacity:")
        self.location_capacity = self._data_editor(
            self.location_capacity, ["MaxCapacity"]
        )

    def _filter_dimensions_class3(self):
        pass

    def _edit_data_class3(self):
        st.write("TransportationCosts:")
        self.transportation_costs = self._data_editor(
            self.transportation_costs, ["Cost"]
        )
