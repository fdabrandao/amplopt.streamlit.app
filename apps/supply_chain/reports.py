import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Reports:
    def __init__(self, problem_number, instance, ampl, key=""):
        self.problem_number = problem_number
        self.instance = instance
        self.ampl = ampl
        self.key = key

    def _planning_view(
        self,
        key,
        df,
        view_func,
        filter_products=False,
        filter_locations=False,
        filter_resources=False,
        filter_customers=False,
        product_location=None,
    ):
        if product_location is not None:
            product, location = product_location
        else:
            if filter_products:
                product = st.selectbox(
                    "Pick the product 👇",
                    [""] + self.instance.selected_products,
                    key=f"{self.key}_{key}_view_product",
                )
            else:
                product = ""

            if filter_locations:
                location = st.selectbox(
                    "Pick the location 👇",
                    [""]
                    + self.instance.locations_with.get(
                        product, self.instance.selected_locations
                    ),
                    key=f"{self.key}_{key}_view_location",
                )
            else:
                location = ""

        if filter_customers:
            customer = st.selectbox(
                "Pick the customer 👇",
                [""]
                + self.instance.customers_buying.get(
                    product, self.instance.all_customers
                ),
                key=f"{self.key}_{key}_view_customer",
            )
        else:
            customer = ""

        if filter_resources:
            resource = st.selectbox(
                "Pick the resource 👇",
                [""]
                + self.instance.resources_at.get(location, self.instance.all_resources),
                key=f"{self.key}_{key}_view_resource",
            )
        else:
            resource = ""

        label = ""
        filter = True
        if location != "" and product != "":
            filter = (df["Location"] == location) & (df["Product"] == product)
            if label == "":
                label = location
            else:
                label = f"{product} at {location}"
        elif product != "" and customer != "":
            filter = (df["Product"] == product) & (df["Customer"] == customer)
            if label == "":
                label = customer
            else:
                label = f"{product} for {customer}"
        elif location != "" and resource != "":
            filter = (df["Location"] == location) & (df["Resource"] == resource)
            if label == "":
                label = location
            else:
                label = f"{resource} at {location}"
        elif location != "":
            filter = df["Location"] == location
            if label == "":
                label = location
        elif customer != "":
            filter = df["Customer"] == customer
            if label == "":
                label = customer
        elif product != "":
            filter = df["Product"] == product
            if label == "":
                label = product
        elif resource != "":
            filter = df["Resource"] == resource
            if label == "":
                label = resource

        if filter is True:
            view_func(df, label)
        else:
            view_func(df[filter], label)

    def demand_report(self, by):
        demand_df = self.ampl.get_data("Demand", "MetDemand", "UnmetDemand").to_pandas()
        demand_df.reset_index(inplace=True)
        if by == "location":
            demand_df.columns = ["Product", "Location", "Period"] + list(
                demand_df.columns[3:]
            )
        elif by == "customer":
            demand_df.columns = ["Product", "Customer", "Period"] + list(
                demand_df.columns[3:]
            )
        else:
            raise ValueError("'by' must be 'location' or 'customer'")

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
                    f"{yval:.1f}",
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
                        f"{yval:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )

            # Show the plot
            st.pyplot(plt)
            # Show the table
            st.dataframe(pivot_table.T)

        if by == "location":
            view = st.selectbox(
                "Demand Report",
                [
                    "Planning View",
                    "Planning View Per Product",
                    "Planning View Per Location",
                    "Full Report",
                ],
            )
        else:
            view = st.selectbox(
                "Demand Report",
                [
                    "Planning View",
                    "Planning View Per Product",
                    "Planning View Per Customer",
                    "Full Report",
                ],
            )

        if view == "Planning View":
            self._planning_view(
                "demand",
                demand_df,
                demand_planning_view,
                filter_products=True,
                filter_locations=(by == "location"),
                filter_customers=(by == "customer"),
            )
        elif view == "Planning View Per Product":
            self._planning_view(
                "demand", demand_df, demand_planning_view, filter_products=True
            )
        elif view == "Planning View Per Location":
            self._planning_view(
                "demand", demand_df, demand_planning_view, filter_locations=True
            )
        elif view == "Planning View Per Customer":
            self._planning_view(
                "demand", demand_df, demand_planning_view, filter_customers=True
            )
        else:
            st.dataframe(demand_df, hide_index=True)

    def resource_utilization_report(self):
        resource_df = self.ampl.get_data(
            "{r in RESOURCES, l in LOCATIONS, t in PERIODS} AvailableCapacity[r,l]",
            "{r in RESOURCES, l in LOCATIONS, t in PERIODS} sum{(p, l) in PRODUCTS_LOCATIONS} ProductionHours[p,l,r,t]",
        ).to_pandas()
        resource_df.reset_index(inplace=True)
        resource_df.columns = [
            "Resource",
            "Location",
            "Period",
            "AvailableCapacity",
            "UsedCapacity",
        ]
        resource_df["UnusedCapacity"] = (
            resource_df["AvailableCapacity"] - resource_df["UsedCapacity"]
        )

        def resource_utilization_planning_view(df, label):
            columns = [
                "AvailableCapacity",
                "UsedCapacity",
                "UnusedCapacity",
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
                df.loc["AvailableCapacity", :],
                label="AvailableCapacity",
                edgecolor="black",
                linewidth=1.5,
                facecolor="none",
            )
            used_bars = ax.bar(
                df.columns,
                df.loc["UsedCapacity", :],
                label="UsedCapacity",
                color="green",
            )
            unused_bars = ax.bar(
                df.columns,
                df.loc["UnusedCapacity", :],
                bottom=df.loc["UsedCapacity", :],
                label="UnusedCapacity",
                color="red",
            )

            # Adding labels and title
            ax.set_ylabel("Units")
            ax.set_title(f"{label} Resource Utilization Overview")
            ax.legend()

            # Adding text inside the bars
            for bar in used_bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{yval:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

            for bar in unused_bars:
                yval = bar.get_height()
                if yval > 0:  # Only display if there's a noticeable unmet demand
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_y() - yval / 2,
                        f"{yval:.1f}",
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
            "Resource Utilization Report",
            [
                "Planning View",
                "Planning View Per Resource",
                "Planning View Per Location",
                "Full Report",
            ],
        )

        if view == "Planning View":
            self._planning_view(
                "resource_utilization",
                resource_df,
                resource_utilization_planning_view,
                filter_resources=True,
                filter_locations=True,
            )
        elif view == "Planning View Per Resource":
            self._planning_view(
                "resource_utilization",
                resource_df,
                resource_utilization_planning_view,
                filter_resources=True,
            )
        elif view == "Planning View Per Location":
            self._planning_view(
                "resource_utilization",
                resource_df,
                resource_utilization_planning_view,
                filter_locations=True,
            )
        else:
            st.dataframe(resource_df, hide_index=True)

    def material_balance_report(
        self,
        include_demand=False,
        product_location=None,
        extra_columns=[],
    ):
        model_parameters = set(self.ampl.get_data("_PARS").to_list())  # FIXME
        model_vars = set(self.ampl.get_data("_VARS").to_list())  # FIXME
        model_entities = model_parameters | model_vars
        columns = [
            "StartingInventory",
            "MetDemand",
            "Production",
            "EndingInventory",
        ] + extra_columns
        if include_demand:
            columns += ["Demand", "UnmetDemand"]
        if "TransfersIn" in model_entities:
            columns += ["TransfersIn", "TransfersOut"]
        if "LostInventory" in model_entities:
            columns = columns + ["LostInventory"]
        columns = list(set(columns))
        order = [
            "Product",
            "Location",
            "Period",
            "StartingInventory",
            "Production",
            "TransfersIn",
            "TransfersOut",
            "Demand",
            "MetDemand",
            "UnmetDemand",
            "LostInventory",
            "EndingInventory",
            "TargetStock",
        ]
        material_df = self.ampl.get_data(*columns).to_pandas()
        material_df.reset_index(inplace=True)
        material_df.columns = ["Product", "Location", "Period"] + list(
            material_df.columns[3:]
        )
        if "TargetStock" in model_entities:
            columns = columns + ["TargetStock"]
            target_stock = self.ampl.get_data(
                "{(p, l) in PRODUCTS_LOCATIONS, t in PERIODS} TargetStock[p, l]"
            ).to_dict()
            material_df["TargetStock"] = [
                target_stock.get((p, l, t), 0)
                for p, l, t in zip(
                    material_df["Product"],
                    material_df["Location"],
                    material_df["Period"],
                )
            ]

        # Reorder the columns
        assert set(columns) - set(order) == set()
        columns = [c for c in order if c in columns]

        def material_balance(df, label):
            pivot_table = pd.pivot_table(
                df,
                index="Period",  # Use 'Period' as the index
                values=columns,  # Specify the columns to aggregate
                aggfunc="sum",  # Use sum as the aggregation function
            )[columns]

            df = pivot_table.T
            fig, ax = plt.subplots(figsize=(12, 4))

            # Plot lines for all columns
            for column in columns:
                ax.plot(
                    df.columns,
                    df.loc[column, :],
                    label=column,
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

        if product_location is not None:
            self._planning_view(
                "material",
                material_df,
                material_balance,
                product_location=product_location,
            )
        else:
            view = st.selectbox(
                "Material Balance Report",
                [
                    "Planning View",
                    "Planning View Per Product",
                    "Planning View Per Location",
                    "Full Report",
                ],
                key=f"{self.key}_material_balance_report_view",
            )

            if view == "Planning View":
                self._planning_view(
                    "material",
                    material_df,
                    material_balance,
                    filter_products=True,
                    filter_locations=True,
                )
            elif view == "Planning View Per Product":
                self._planning_view(
                    "material", material_df, material_balance, filter_products=True
                )
            elif view == "Planning View Per Location":
                self._planning_view(
                    "material", material_df, material_balance, filter_locations=True
                )
            else:
                st.dataframe(material_df, hide_index=True)

    def network_report(self):
        model_vars = set(self.ampl.get_data("_VARS").to_list())  # FIXME
        product = st.selectbox(
            "Pick the product 👇",
            [""] + self.instance.selected_products,
            key=f"{self.key}_transfers_view_product",
        )
        periods = self.ampl.set["PERIODS"].to_list()
        period = st.selectbox(
            "Pick the time period 👇",
            periods,
            key=f"{self.key}_transfers_view_period",
        )

        nodes = self.ampl.set["LOCATIONS"].to_list()
        edge_labels = {}
        if "Transfers" in model_vars:
            all_transfers = self.ampl.var["Transfers"].to_dict()
            transfers = {}
            for (p, i, j, t), value in all_transfers.items():
                if t != period or value <= 1e-5:
                    continue
                if p == product or product == "":
                    if (i, j) not in transfers:
                        transfers[i, j] = 0
                    transfers[i, j] += value

            edge_labels = {
                (i, j): f"{value:.2f}" for (i, j), value in transfers.items()
            }

        if self.problem_number == 2:
            all_shipments = self.ampl.var["Shipments"].to_dict()
            shipments = {}
            for (p, i, j, t), value in all_shipments.items():
                if t != period or value <= 1e-5:
                    continue
                if p == product or product == "":
                    if (i, j) not in shipments:
                        shipments[i, j] = 0
                    shipments[i, j] += value

            edge_labels.update(
                {(i, j): f"{value:.2f}" for (i, j), value in shipments.items()}
            )

        G = nx.DiGraph()
        G.add_nodes_from(sorted(nodes))
        G.add_edges_from(sorted(edge_labels.keys()))
        pos = nx.circular_layout(G)

        node_colors = [
            "yellow" if node in self.instance.selected_customers else "orange"
            for node in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.margins(0.15)

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=3000,
            font_size=10,
            # labels=node_labels,
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        st.pyplot(fig)
        pass
