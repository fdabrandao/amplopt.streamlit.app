import streamlit as st
import re


class ModelBuilder:
    def __init__(self, class_number, use_restrict_table, show_complete_model):
        self.class_number = class_number
        self.use_restrict_table = use_restrict_table
        self.show_complete_model = show_complete_model
        if class_number == 1:
            self.model = self.base_model()

            self.model += r"""
            ##################
            # Demand Balance # 
            ##################
            """
            self.model += self.demand_fulfillment_declaration()

            self.model += r"""
            #######################
            # Inventory Carryover # 
            #######################
            """
            self.model += self.inventory_carryover_declaration()

            self.model += r"""
            ####################
            # Material Balance # 
            ####################
            """
            self.model += self.material_balance_declaration()

            self.model += r"""
            #############
            # Objective #
            #############
            """
            self.model += self.class1_objective()
        elif class_number == 2:
            self.model = self.base_model()

            self.model += r"""
            ##################
            # Demand Balance # 
            ##################
            """
            self.model += self.demand_fulfillment_declaration(show=True)

            self.model += r"""
            #######################
            # Inventory Carryover # 
            #######################
            """
            self.model += self.inventory_carryover_declaration(show=True)

            self.model += r"""
            ###########################################
            # Part 1: Production and Production Hours #
            ###########################################
            """
            self.model += self.production_rate_declaration()

            self.model += r"""
            #############################
            # Part 2: Resource capacity #
            #############################
            """
            self.model += self.resource_capacity_declaration()

            self.model += r"""
            #####################
            # Part 3: Transfers #
            #####################
            """
            self.model += self.material_balance_with_transfers_declaration()

            self.model += r"""
            #########################
            # Part 4: Target Stocks # 
            #########################
            """
            self.model += self.target_stock_declaration()

            self.model += r"""
            ############################
            # Part 5: Storage Capacity #
            ############################
            """
            self.model += self.storage_capacity_declaration()

            self.model += r"""
            #############
            # Objective #
            #############
            """
            self.model += self.class2_objective()
        else:
            assert False

    def _transform(self, declaration):
        if self.use_restrict_table:
            return declaration.replace(
                "p in PRODUCTS, l in LOCATIONS", "(p, l) in PRODUCTS_LOCATIONS"
            )
        return declaration

    def _exercise(
        self, ampl, name, constraint, needs, allow_skipping=True, skip=False, help=""
    ):
        if skip or (
            allow_skipping
            and st.checkbox(f"Skip exercise", key=f"Skip {name}", value=True)
        ):
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
                # output = ampl.get_output("write 0;")
                # if output != "" and not output.startswith("No files written"):
                #     if "Error executing " in output:
                #         output = output[output.find(":") + 1 :].strip()
                #     st.error(f"❌ Error: {output}")

    def base_model(self):
        return self._transform(
            r"""
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
            """
        )

    def demand_fulfillment_declaration(self, show=None):
        self.demand_fulfillment = self._transform(
            """
            s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                Demand[p, l, t] = MetDemand[p, l, t] + UnmetDemand[p, l, t];
                # Ensure that all demand is accounted for either as met or unmet
            """
        )

        self.demand_fulfillment_placeholder = self._transform(
            r"""
            # s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 1: Ensure that all demand is accounted for either as met or unmet
            """
        )

        if show or self.show_complete_model:
            return self.demand_fulfillment
        else:
            return self.demand_fulfillment_placeholder

    def _skip_flag(self, selected_exercise, exercise_number):
        allow_skipping = selected_exercise != exercise_number
        skip = selected_exercise != 0 and (
            selected_exercise == -1 or selected_exercise != exercise_number
        )
        return allow_skipping, skip

    def demand_fulfillment_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 1)
        if not skip:
            st.markdown(
                """
                ### Exercise #1: Demand Balance Constraint
                
                🧑‍🏫 Ensure that all demand is accounted for either as met or unmet.
                """
            )
        self._exercise(
            ampl,
            "Demand Balance Constraint",
            self.demand_fulfillment,
            ["Demand[p, l, t]", "MetDemand[p, l, t]", "UnmetDemand[p, l, t]", "="],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def inventory_carryover_declaration(self, show=None):
        self.inventory_carryover = self._transform(
            r"""
            s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p, l, t] =
                    if ord(t) > 1 then
                        EndingInventory[p, l, prev(t)]
                    else
                        InitialInventory[p, l];
                # Define how inventory is carried over from one period to the next
            """
        )

        self.inventory_carryover_placeholder = self._transform(
            r"""
            # s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 2: Define how inventory is carried over from one period to the next
            """
        )

        if show or self.show_complete_model:
            return self.inventory_carryover
        else:
            return self.inventory_carryover_placeholder

    def inventory_carryover_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 2)
        if not skip:
            st.markdown(
                """
                ### Exercise #2: Inventory Carryover Constraint
                
                🧑‍🏫 Define how inventory is carried over from one period to the next.
                """
            )
        self._exercise(
            ampl,
            "Inventory Carryover Constraint",
            self.inventory_carryover,
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
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def material_balance_declaration(self, show=None):
        self.material_balance = self._transform(
            r"""
            s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p, l, t] + Production[p, l, t] - MetDemand[p, l, t] = EndingInventory[p, l, t];
                # Balance starting inventory and production against demand to determine ending inventory
            """
        )

        self.material_balance_placeholder = self._transform(
            r"""
            # s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 3: Balance starting inventory and production against demand to determine ending inventory
            """
        )

        if show or self.show_complete_model:
            return self.material_balance
        else:
            return self.material_balance_placeholder

    def material_balance_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 3)
        if not skip:
            st.markdown(
                """
                ### Exercise #3: Material Balance Constraint
                
                🧑‍🏫 Balance starting inventory and production against demand to determine ending inventory.
                """
            )
        self._exercise(
            ampl,
            "Material Balance Constraint",
            self.material_balance,
            [
                "StartingInventory[p, l, t]",
                "Production[p, l, t]",
                "MetDemand[p, l, t]",
                "EndingInventory[p, l, t]",
                "=",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def production_rate_declaration(self, show=None):
        header = self._transform(
            r"""
            set RESOURCES;  # Set of production resources
            
            var ProductionHours{p in PRODUCTS, l in LOCATIONS, r in RESOURCES, t in PERIODS} >= 0; 
                # Production hours for each product, location, resource, and period
            param ProductionRate{p in PRODUCTS, l in LOCATIONS, r in RESOURCES} >= 0 default 0;
                # Production rate for each product at each location and resource
            """
        )

        self.production_rate = self._transform(
            r"""
            # Exercise 1
            s.t. ProductionRateConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                Production[p,l,t] == sum{r in RESOURCES} ProductionHours[p,l,r,t] * ProductionRate[p,l,r];
                # Ensure that the total production quantity is equal to the production hours multiplied by the production rate
            """
        )

        self.production_rate_placeholder = self._transform(
            r"""
            # s.t. ProductionRateConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 1: Ensure that the total production quantity is equal to the production hours multiplied by the production rate
            """
        )

        if show or self.show_complete_model:
            return header + self.production_rate
        else:
            return header + self.production_rate_placeholder

    def production_rate_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 1)
        if not skip:
            st.markdown(
                """
                ### Exercise #1: Production and Production Hours
                
                🧑‍🏫 Ensure that the total production quantity is equal to the production hours multiplied by the production rate.
                """
            )
        self._exercise(
            ampl,
            "Production and Production Hours",
            self.production_rate,
            [
                "Production[p,l,t]",
                "sum{r in RESOURCES}",
                "ProductionHours[p,l,r,t]",
                "*",
                "ProductionRate[p,l,r]",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def resource_capacity_declaration(self, show=None):
        header = self._transform(
            r"""
            param AvailableCapacity{r in RESOURCES, l in LOCATIONS} >= 0 default 0; 
                # Available capacity for each resource at each location
            """
        )

        self.resource_capacity = self._transform(
            r"""
            # Exercise 2
            s.t. ProductionCapacity{r in RESOURCES, l in LOCATIONS, t in PERIODS}:
                sum{(p, l) in PRODUCTS_LOCATIONS} ProductionHours[p,l,r,t] <= AvailableCapacity[r,l];
                # Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location
            """
        )

        self.resource_capacity_placeholder = self._transform(
            r"""
            # s.t. ProductionCapacity{r in RESOURCES, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 2: Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location
            """
        )

        if show or self.show_complete_model:
            return header + self.resource_capacity
        else:
            return header + self.resource_capacity_placeholder

    def resource_capacity_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 2)
        if not skip:
            st.markdown(
                """
                ### Exercise #2: Resource capacity
                
                🧑‍🏫 Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location.
                """
            )
        self._exercise(
            ampl,
            "Resource capacity",
            self.resource_capacity,
            [
                "sum{(p, l) in PRODUCTS_LOCATIONS}",
                "ProductionHours[p,l,r,t]",
                "<=",
                "AvailableCapacity[r,l]",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def material_balance_with_transfers_declaration(self, show=None):
        header = self._transform(
            r"""
            set TRANSFER_LANES within {PRODUCTS, LOCATIONS, LOCATIONS};
                # Valid transfer lanes (From_Location, To_Location)
            var TransfersIN{(p, i, j) in TRANSFER_LANES, t in PERIODS} >= 0;
                # Transfers of product 'p' arriving at location 'j' from location 'i'
            var TransfersOUT{(p, i, j) in TRANSFER_LANES, t in PERIODS} >= 0;
                # Transfers of product 'p' leaving from location 'i' to location 'j'
            """
        )

        self.material_balance_with_transfers = self._transform(
            r"""
            # Exercise 3:
            s.t. MaterialBalanceWithTransfers{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p,l,t] - MetDemand[p,l,t] + Production[p,l,t]
                + sum{i in LOCATIONS: (p, i, l) in TRANSFER_LANES} TransfersIN[p,i,l,t]
                - sum{j in LOCATIONS: (p, l, j) in TRANSFER_LANES} TransfersOUT[p,l,j,t]
                == EndingInventory[p,l,t];
                # Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment
            """
        )

        self.material_balance_with_transfers_placeholder = self._transform(
            r"""
            # s.t. MaterialBalanceWithTransfers{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 3: Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment
            """
        )

        if show or self.show_complete_model:
            return header + self.material_balance_with_transfers
        else:
            return header + self.material_balance_with_transfers_placeholder

    def material_balance_with_transfers_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 3)
        if not skip:
            st.markdown(
                """
                ### Exercise #3: Transfers
                
                🧑‍🏫 Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment.
                """
            )
        self._exercise(
            ampl,
            "Transfers",
            self.material_balance_with_transfers,
            [
                "StartingInventory[p,l,t]",
                "MetDemand[p,l,t]",
                "Production[p,l,t]",
                "sum{i in LOCATIONS: (p, i, l) in TRANSFER_LANES}",
                "TransfersIN[p,i,l,t]",
                "sum{j in LOCATIONS: (p, l, j) in TRANSFER_LANES}",
                "TransfersOUT[p,l,j,t]",
                "EndingInventory[p,l,t]",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def target_stock_declaration(self, show=None):
        header = self._transform(
            r"""
            param TargetStock{p in PRODUCTS, l in LOCATIONS} >= 0 default 0;
                # Target stock level for each product and location
            var AboveTarget{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Amount above target stock
            var BelowTarget{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Amount below target stock
            """
        )

        self.target_stock = self._transform(
            r"""
            # Exercise 4:
            s.t. TargetStockConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                TargetStock[p, l] == EndingInventory[p, l, t] + BelowTarget[p, l, t] - AboveTarget[p, l, t];
                # Ensure that the ending inventory is adjusted to either exceed (AboveTarget) or fall below (BelowTarget) the target stock level
            """
        )

        self.target_stock_placeholder = self._transform(
            r"""
            # s.t. TargetStockConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... Exercise 4: Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """
        )

        if show or self.show_complete_model:
            return header + self.target_stock
        else:
            return header + self.target_stock_placeholder

    def target_stock_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 4)
        if not skip:
            st.markdown(
                """
                ### Exercise #4: Target Stocks
                
                🧑‍🏫 Ensure that the ending inventory is adjusted to either exceed (AboveTarget) or fall below (BelowTarget) the target stock level.
                """
            )
        self._exercise(
            ampl,
            "Target Stocks",
            self.target_stock,
            [
                "TargetStock[p, l]",
                "EndingInventory[p, l, t]",
                "BelowTarget[p, l, t]",
                "AboveTarget[p, l, t]",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def storage_capacity_declaration(self, show=None):
        header = self._transform(
            r"""
            param MaxCapacity{l in LOCATIONS} >= 0;
                # Maximum storage capacity for each location and period
            """
        )

        self.storage_capacity = self._transform(
            r"""
            # Exercise 5:
            subject to StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
                sum{(p, l) in PRODUCTS_LOCATIONS} EndingInventory[p, l, t] <= MaxCapacity[l];
                # Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """
        )

        self.storage_capacity_placeholder = self._transform(
            r"""
            # s.t. StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
            # ... Exercise 5: Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """
        )

        if show or self.show_complete_model:
            return header + self.storage_capacity
        else:
            return header + self.storage_capacity_placeholder

    def storage_capacity_exercise(self, ampl, selected_exercise):
        allow_skipping, skip = self._skip_flag(selected_exercise, 5)
        if not skip:
            st.markdown(
                """
                ### Exercise #5: Storage Capacity
                
                🧑‍🏫 Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location.
                """
            )
        self._exercise(
            ampl,
            "Storage Capacity",
            self.storage_capacity,
            [
                "sum{(p, l) in PRODUCTS_LOCATIONS}",
                "EndingInventory[p, l, t]",
                "<=",
                "MaxCapacity[l]",
            ],
            allow_skipping=allow_skipping,
            skip=skip,
        )

    def class1_objective(self):
        return self._transform(
            r"""
            param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
            param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)

            minimize TotalCost:
                sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS}
                    (UnmetDemandPenalty * UnmetDemand[p, l, t] + EndingInventoryPenalty * EndingInventory[p, l, t]);
                # Objective function to minimize total costs associated with unmet demand and leftover inventory
            """
        )

    def class2_objective(self):
        return self._transform(
            r"""
            param BelowTargetPenalty default 3;
                # Penalty for having inventory below target
            param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
            param AboveTargetPenalty default 2;
                # Penalty for having inventory above target
            param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)
            param TransferPenalty default 1;
                # Penalty for each unit transferred

            # Minimize total cost objective
            minimize TotalCost:
                sum{p in PRODUCTS, l in LOCATIONS, t in PERIODS} (
                    UnmetDemandPenalty * UnmetDemand[p, l, t] 
                    + EndingInventoryPenalty * EndingInventory[p, l, t] 
                    + AboveTargetPenalty * AboveTarget[p, l, t] 
                    + BelowTargetPenalty * BelowTarget[p, l, t]
                )
                + sum{(p, i, j) in TRANSFER_LANES, t in PERIODS} (
                    TransferPenalty * TransfersOUT[p, i, j, t]
                );
                # Objective: Minimize total cost, which includes penalties for unmet demand, ending inventory, deviations from target stock, and transfers
            """
        )
