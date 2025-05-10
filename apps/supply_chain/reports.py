import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


class Reports:
    def __init__(self, instance, ampl):
        self.instance = instance
        self.ampl = ampl

    def _planning_view(
        self,
        key,
        df,
        view_func,
        filter_products=False,
        filter_locations=False,
        filter_resources=False,
    ):
        if filter_products:
            product = st.selectbox(
                "Pick the product 👇",
                [""] + self.instance.selected_products,
                key=f"{key}_view_product",
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
                key=f"{key}_view_location",
            )
        else:
            location = ""

        if filter_resources:
            resource = st.selectbox(
                "Pick the resource 👇",
                [""]
                + self.instance.resources_at.get(location, self.instance.all_resources),
                key=f"{key}_view_resource",
            )
        else:
            resource = ""

        label = ""
        filter = True
        if product != "":
            filter = df["Product"] == product
            label = product
        if resource != "":
            filter = df["Resource"] == resource
            label = resource

        if location != "" and product != "":
            filter = (df["Location"] == location) & filter
            if label == "":
                label = location
            else:
                label = f"{product} at {location}"
        elif location != "" and resource != "":
            filter = (df["Location"] == location) & filter
            if label == "":
                label = location
            else:
                label = f"{resource} at {location}"
        elif location != "":
            filter = (df["Location"] == location) & filter
            if label == "":
                label = location

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
            self._planning_view(
                "demand",
                demand_df,
                demand_planning_view,
                filter_products=True,
                filter_locations=True,
            )
        elif view == "Planning View Per Product":
            self._planning_view(
                "demand", demand_df, demand_planning_view, filter_products=True
            )
        elif view == "Planning View Per Location":
            self._planning_view(
                "demand", demand_df, demand_planning_view, filter_locations=True
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
        include_transfers=False,
        include_target_stock=False,
        include_shelf_life=True,
    ):
        columns = ["StartingInventory", "MetDemand", "Production", "EndingInventory"]
        material_df = self.ampl.get_data(*columns).to_pandas()
        material_df.reset_index(inplace=True)
        material_df.columns = ["Product", "Location", "Period"] + list(
            material_df.columns[3:]
        )
        if include_shelf_life:
            columns = columns + ["LostInventory"]
            lost = self.ampl.get_data("LostInventory").to_dict()
            material_df["LostInventory"] = [
                lost.get((p, l, t), 0)
                for p, l, t in zip(
                    material_df["Product"],
                    material_df["Location"],
                    material_df["Period"],
                )
            ]
            columns.remove("EndingInventory")
            columns.append("EndingInventory")
        if include_transfers:
            columns = columns + ["TransfersIN", "TransfersOUT"]
            transfers_in = self.ampl.get_data("TransfersIN").to_dict()
            transfers_out = self.ampl.get_data("TransfersOUT").to_dict()
            material_df["TransfersIN"] = [
                transfers_in.get((p, l, t), 0)
                for p, l, t in zip(
                    material_df["Product"],
                    material_df["Location"],
                    material_df["Period"],
                )
            ]
            material_df["TransfersOUT"] = [
                transfers_out.get((p, l, t), 0)
                for p, l, t in zip(
                    material_df["Product"],
                    material_df["Location"],
                    material_df["Period"],
                )
            ]
            columns.remove("EndingInventory")
            columns.append("EndingInventory")
        if include_target_stock:
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

            if include_transfers:
                ax.plot(
                    df.columns,
                    df.loc["TransfersIN", :],
                    label="TransfersIN",
                    marker="o",
                )
                ax.plot(
                    df.columns,
                    df.loc["TransfersOUT", :],
                    label="TransfersOUT",
                    marker="o",
                )

            if include_target_stock:
                ax.plot(
                    df.columns,
                    df.loc["TargetStock", :],
                    label="TargetStock",
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
