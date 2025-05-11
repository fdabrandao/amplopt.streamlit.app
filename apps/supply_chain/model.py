import streamlit as st
import textwrap
import re


class Exercise:
    def __init__(self, number, name, render_exercise):
        self.number = number
        self.name = name
        self.render_exercise = render_exercise

    def title(self):
        return f"Exercise #{self.number}: {self.name}"

    def render(self, ampl, selected_exercise):
        self.render_exercise(
            ampl,
            name=self.name,
            number=self.number,
            selected_exercise=selected_exercise,
        )


class Parameter:
    def __init__(self, name, render_controller):
        self._name = name
        self.render_controller = render_controller

    def name(self):
        return self._name

    def render(self):
        return self.render_controller()


class ModelBuilder:
    def __init__(
        self,
        class_number,
        use_restrict_table,
        show_complete_model,
        model_shelf_life,
        layered_storage_capacity,
        layered_targets,
        model_incremental_lot_sizing,
        lot_sizing_mp,
        include_homework3,
        on_change=None,
    ):
        self.model = ""
        self.exercises = []
        self.parameters = []
        self.on_change = on_change
        self.class_number = class_number
        self.use_restrict_table = use_restrict_table
        self.show_complete_model = show_complete_model
        self.model_shelf_life = model_shelf_life
        self.layered_storage_capacity = layered_storage_capacity
        self.layered_targets = layered_targets
        if class_number == 1:
            self.add_base_model()

            self.add(
                r"""
            ##################
            # Demand Balance # 
            ##################
                """
            )
            self.add_demand_fulfillment_declaration(exercise=1)

            self.add(
                r"""
            #######################
            # Inventory Carryover # 
            #######################
                """
            )
            self.add_inventory_carryover_declaration(exercise=2)

            self.add(
                r"""
            ####################
            # Material Balance # 
            ####################
                """
            )
            self.add_material_balance_declaration(exercise=3)

            self.add(
                r"""
            #############
            # Objective #
            #############
                """
            )
            self.add_class1_objective()
        elif class_number == 2:
            if not self.model_shelf_life:
                self.add_base_model()
            else:
                self.add_base_model_with_shelf_life()

            self.add(
                r"""
            ##################
            # Demand Balance # 
            ##################
                """
            )
            self.add_demand_fulfillment_declaration(show=True)

            inventory_carryover_header = r"""
            #######################
            # Inventory Carryover # 
            #######################
            """

            inventory_carryover_with_shelf_life_header = r"""
            #######################################
            # Inventory Carryover With Shelf-Life # 
            #######################################
            """

            if not self.model_shelf_life:
                self.add(inventory_carryover_header)
                self.add_inventory_carryover_declaration(show=True)
            else:
                self.add(inventory_carryover_with_shelf_life_header)
                self.add_inventory_carryover_with_shelf_life_declaration(show=True)

            material_balance_header = r"""
            ####################
            # Material Balance # 
            ####################
            """

            material_balance_with_shelf_life_header = r"""
            ####################################
            # Material Balance With Shelf-Life # 
            ####################################
            """

            if not self.model_shelf_life:
                self.add(material_balance_header)
                self.add_material_balance_declaration(show=True)
            else:
                self.add(material_balance_with_shelf_life_header)
                self.add_material_balance_with_shelf_life_declaration(show=True)

            self.add(
                r"""
            ###########################################
            # Part 1: Production and Production Hours #
            ###########################################
                """
            )
            self.add_production_rate_declaration(exercise=1)

            self.add(
                r"""
            #############################
            # Part 2: Resource capacity #
            #############################
                """
            )
            self.add_resource_capacity_declaration(exercise=2)

            self.add(
                r"""
            #############
            # Objective #
            #############
                """
            )
            if not self.model_shelf_life:
                self.add_class1_objective()
            else:
                self.add_class1_objective_with_shelf_life()
        elif class_number == 3:
            self.add_base_model()

            self.add(
                r"""
            ##################
            # Demand Balance # 
            ##################
                """
            )
            self.add_demand_fulfillment_declaration(show=True)

            self.add(
                r"""
            #######################
            # Inventory Carryover # 
            #######################
                """
            )
            self.add_inventory_carryover_declaration(show=True)

            self.add(
                r"""
            ###################################
            # Production and Production Hours #
            ###################################
                """
            )
            self.add_production_rate_declaration(show=True)

            self.add(
                r"""
            #####################
            # Resource capacity #
            #####################
                """
            )
            self.add_resource_capacity_declaration(show=True)

            self.add(
                r"""
            #####################
            # Part 1: Transfers #
            #####################
                """
            )
            self.add_material_balance_with_transfers_declaration(exercise=1)

            self.add(
                r"""
            #########################
            # Part 2: Target Stocks # 
            #########################
                """
            )
            self.add_target_stock_declaration(exercise=2)

            self.add(
                r"""
            ############################
            # Part 3: Storage Capacity #
            ############################
                """
            )
            if not self.layered_storage_capacity:
                self.add_storage_capacity_declaration(exercise=3)
            else:
                self.add_soft_storage_capacity_declaration(exercise=3)

            self.add(
                r"""
            #############
            # Objective #
            #############
                """
            )
            self.add_class3_objective(
                self.layered_storage_capacity, self.layered_targets
            )
        elif class_number == 4:
            self.add_base_model()

            self.add(
                r"""
            ##################
            # Demand Balance # 
            ##################
                """
            )
            self.add_demand_fulfillment_declaration(show=True)

            self.add(
                r"""
            #######################
            # Inventory Carryover # 
            #######################
                """
            )
            self.add_inventory_carryover_declaration(show=True)

            self.add(
                r"""
            ###################################
            # Production and Production Hours #
            ###################################
                """
            )
            self.add_production_rate_declaration(show=True)

            self.add(
                r"""
            #####################
            # Resource capacity #
            #####################
                """
            )
            self.add_resource_capacity_declaration(show=True)

            if include_homework3:
                self.add(
                    r"""
                #############
                # Transfers #
                #############
                    """
                )
                self.add_material_balance_with_transfers_declaration(show=True)

                self.add(
                    r"""
                #################
                # Target Stocks # 
                #################
                    """
                )
                self.add_target_stock_declaration(show=True)

                self.add(
                    r"""
                ####################
                # Storage Capacity #
                ####################
                    """
                )
                self.add_storage_capacity_declaration(show=True)

            if not model_incremental_lot_sizing:
                self.add(
                    r"""
                    ##############
                    # Lot-Sizing #
                    ##############
                    """
                )
                self.add_lot_sizing_min(
                    use_mp=lot_sizing_mp,
                    exercise=1,
                )
            else:
                self.add(
                    r"""
                    ##########################
                    # Incremental Lot-Sizing #
                    ##########################
                    """
                )
                self.add_lot_sizing_incremental(
                    use_mp=lot_sizing_mp,
                    exercise=1,
                )

            self.add(
                r"""
            #############
            # Objective #
            #############
                """
            )
            if include_homework3:
                self.add_class3_objective(
                    self.layered_storage_capacity, self.layered_targets
                )
            else:
                self.add_class1_objective()
        else:
            assert False

    def add(self, declaration, transform=False):
        if transform:
            declaration = self._transform(declaration)
        self.model += textwrap.dedent(declaration)

    def _transform(self, declaration, exercise=None):
        declaration = textwrap.dedent(declaration)
        if self.use_restrict_table:
            declaration = declaration.replace(
                "p in PRODUCTS, l in LOCATIONS", "(p, l) in PRODUCTS_LOCATIONS"
            )
        if exercise is not None:
            assert "!exercise!" in declaration
            declaration = declaration.replace(
                "... !exercise!",
                f"Exercise #{exercise}: ",
            )
            declaration = declaration.replace(
                "!exercise!",
                f"# Exercise #{exercise}: ",
            )
        else:
            declaration = declaration.replace("!exercise!", "")

        return re.sub(r"\n\s*\n*\s*\n", "\n", declaration).replace("!empty!", "")

    def _skip_flag(self, selected_exercise, exercise_number):
        allow_skipping = selected_exercise != exercise_number
        skip = selected_exercise != 0 and (
            selected_exercise == -1 or selected_exercise != exercise_number
        )
        return allow_skipping, skip

    def _exercise(
        self,
        ampl,
        name,
        description,
        exercise,
        selected_exercise,
        constraint,
        needs,
        help="",
    ):
        allow_skipping, skip = self._skip_flag(selected_exercise, exercise)
        if not skip:
            st.markdown(
                f"""
                ### !exercise!{name}
                
                🧑‍🏫 {description}
                """.replace(
                    "!exercise!",
                    f"Exercise #{exercise}: " if exercise is not None else "",
                )
            )
        if skip or (
            allow_skipping
            and st.checkbox(
                f"Skip exercise",
                key=f"Skip {name}",
                value=True,
                on_change=self.on_change,
            )
        ):
            ampl.eval(constraint)
        else:
            constraint = constraint[constraint.find("s.t.") :]
            constraint = constraint[: constraint.find("\n")] + "\n\t"
            answer = st.text_input(
                f"Implement the {name} below",
                on_change=self.on_change,
            ).strip()
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

    def add_base_model(self):
        self.add(
            r"""
            set PRODUCTS;  # Set of products
            set LOCATIONS;  # Set of distribution or production locations
            set PRODUCTS_LOCATIONS within {PRODUCTS, LOCATIONS};  # Restrict table
            set PERIODS ordered;  # Ordered set of time periods for planning
            !empty!
            param Demand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0 default 0;
                # Demand for each product at each location during each time period
            var UnmetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Quantity of demand that is not met for a product at a location in a time period
            var MetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Quantity of demand that is met for a product at a location in a time period
            !empty!
            param InitialInventory{p in PRODUCTS, l in LOCATIONS} >= 0 default 0;
                # Initial inventory levels for each product at each location
            var StartingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Inventory at the beginning of each time period
            var EndingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Inventory at the end of each time period
            var Production{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Production volume for each product at each location during each time period
            """,
            transform=True,
        )

    def add_base_model_with_shelf_life(self):
        self.add(
            r"""
            set PRODUCTS;  # Set of products
            set LOCATIONS;  # Set of distribution or production locations
            set PRODUCTS_LOCATIONS within {PRODUCTS, LOCATIONS};  # Restrict table
            set PERIODS ordered;  # Ordered set of time periods for planning
            param MaxShelfLife >= 0;  # Maximum shelf-life of products
            set SHELF_LIFE ordered := 0..MaxShelfLife; # Define the set of shelf-life periods
            !empty!
            param Demand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0 default 0;
                # Demand for each product at each location during each time period
            var UnmetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Quantity of demand that is not met for a product at a location in a time period
            !empty!
            var MetDemandSL{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE} >= 0;
                # Quantity of demand that is met for each product-location-period-shelf-life combination
            var MetDemand{p in PRODUCTS, l in LOCATIONS, t in PERIODS} = sum {d in SHELF_LIFE} MetDemandSL[p, l, t, d];
                # Quantity of demand that is met for a product at a location in a time period
            !empty! 
            param InitialInventory{p in PRODUCTS, l in LOCATIONS} >= 0 default 0;
                # Initial inventory levels for each product at each location
            !empty!
            var StartingInventorySL{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE} >= 0;
                # Inventory at the beginning of each time period for each shelf-life
            var StartingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} = sum {d in SHELF_LIFE} StartingInventorySL[p, l, t, d];
                # Inventory at the beginning of each time period
            !empty!
            var EndingInventorySL{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE} >= 0;
                # Inventory at the end of each time period
            var LostInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} = EndingInventorySL[p, l, t, last(SHELF_LIFE)];
                # Inventory that is lost due to expiration
            var EndingInventory{p in PRODUCTS, l in LOCATIONS, t in PERIODS} = sum {d in SHELF_LIFE: d < last(SHELF_LIFE)} EndingInventorySL[p, l, t, d];
                # Inventory at the end of each time period
            !empty!
            var Production{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Production volume for each product at each location during each time period
            """,
            transform=True,
        )

    def add_demand_fulfillment_declaration(self, exercise=None, show=None):
        demand_fulfillment = self._transform(
            """
            !exercise!
            s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                Demand[p, l, t] = MetDemand[p, l, t] + UnmetDemand[p, l, t];
                # Ensure that all demand is accounted for either as met or unmet
            """,
            exercise=exercise,
        )

        demand_fulfillment_placeholder = self._transform(
            r"""
            # s.t. DemandBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that all demand is accounted for either as met or unmet
            """,
            exercise=exercise,
        )

        exercise_name = "Demand Balance Constraint"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that all demand is accounted for either as met or unmet.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=demand_fulfillment,
                needs=[
                    "Demand[p, l, t]",
                    "MetDemand[p, l, t]",
                    "UnmetDemand[p, l, t]",
                    "=",
                ],
            )

        if show or self.show_complete_model:
            self.add(demand_fulfillment)
        else:
            self.add(demand_fulfillment_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_inventory_carryover_declaration(self, exercise=None, show=None):
        inventory_carryover = self._transform(
            r"""
            !exercise!
            s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p, l, t] =
                    if ord(t) > 1 then
                        EndingInventory[p, l, prev(t)]
                    else
                        InitialInventory[p, l];
                # Define how inventory is carried over from one period to the next
            """,
            exercise=exercise,
        )

        inventory_carryover_placeholder = self._transform(
            r"""
            # s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Define how inventory is carried over from one period to the next
            """,
            exercise=exercise,
        )

        exercise_name = "Inventory Carryover Constraint"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Define how inventory is carried over from one period to the next.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=inventory_carryover,
                needs=[
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

        if show or self.show_complete_model:
            self.add(inventory_carryover)
        else:
            self.add(inventory_carryover_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_inventory_carryover_with_shelf_life_declaration(
        self, exercise=None, show=None
    ):
        inventory_carryover = self._transform(
            r"""
            !exercise!
            s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE}:
                StartingInventorySL[p, l, t, d] =
                    if ord(t) > 1 then
                        (if ord(d) > 1 then EndingInventorySL[p, l, prev(t), prev(d)] else 0)
                    else
                        (if ord(d) = 2 then InitialInventory[p, l] else 0);
                # Define how inventory is carried over from one period to the next
            """,
            exercise=exercise,
        )

        inventory_carryover_placeholder = self._transform(
            r"""
            # s.t. InventoryCarryover{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE}:
            # ... !exercise!Define how inventory is carried over from one period to the next
            """,
            exercise=exercise,
        )

        if show or self.show_complete_model:
            self.add(inventory_carryover)
        else:
            self.add(inventory_carryover_placeholder)

    def add_material_balance_declaration(self, exercise=None, show=None):
        material_balance = self._transform(
            r"""
            !exercise!
            s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p, l, t] + Production[p, l, t] - MetDemand[p, l, t] = EndingInventory[p, l, t];
                # Balance starting inventory and production against demand to determine ending inventory
            """,
            exercise=exercise,
        )

        material_balance_placeholder = self._transform(
            r"""
            # s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Balance starting inventory and production against demand to determine ending inventory
            """,
            exercise=exercise,
        )

        exercise_name = "Material Balance Constraint"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Balance starting inventory and production against demand to determine ending inventory.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=material_balance,
                needs=[
                    "StartingInventory[p, l, t]",
                    "Production[p, l, t]",
                    "MetDemand[p, l, t]",
                    "EndingInventory[p, l, t]",
                    "=",
                ],
            )

        if show or self.show_complete_model:
            self.add(material_balance)
        else:
            self.add(material_balance_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_material_balance_with_shelf_life_declaration(
        self, exercise=None, show=None
    ):
        material_balance = self._transform(
            r"""
            !exercise!
            s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE}:
                StartingInventorySL[p, l, t, d]
                + (if ord(d) == 1 then Production[p, l, t]) 
                - MetDemandSL[p, l, t, d] = EndingInventorySL[p, l, t, d];
                # Balance starting inventory and production against demand to determine ending inventory
            !empty!
            param EnsureOldStockGoesFirst default 1;
            s.t. SellOldStockFirst{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE: EnsureOldStockGoesFirst == 1}:
                EndingInventorySL[p, l, t, d] > 0 ==> sum {dd in SHELF_LIFE: ord(dd) < ord(d)} MetDemandSL[p, l, t, dd] = 0;
                # If there is old inventory, then there should be no demand met for the same product with a shorter shelf-life
            """,
            exercise=exercise,
        )

        material_balance_placeholder = self._transform(
            r"""
            # s.t. MaterialBalance{p in PRODUCTS, l in LOCATIONS, t in PERIODS, d in SHELF_LIFE}:
            # ... !exercise!Balance starting inventory and production against demand to determine ending inventory
            """,
            exercise=exercise,
        )

        if show or self.show_complete_model:
            self.add(material_balance)
        else:
            self.add(material_balance_placeholder)

    def add_production_rate_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            set RESOURCES;  # Set of production resources
            
            var ProductionHours{p in PRODUCTS, l in LOCATIONS, r in RESOURCES, t in PERIODS} >= 0; 
                # Production hours for each product, location, resource, and period
            param ProductionRate{p in PRODUCTS, l in LOCATIONS, r in RESOURCES} >= 0 default 0;
                # Production rate for each product at each location and resource (could also depend on the period)
            """
        )

        production_rate = self._transform(
            r"""
            !exercise!
            s.t. ProductionRateConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                Production[p,l,t] = sum{r in RESOURCES} ProductionHours[p,l,r,t] * ProductionRate[p,l,r];
                # Ensure that the total production quantity is equal to the production hours multiplied by the production rate
            """,
            exercise=exercise,
        )

        production_rate_placeholder = self._transform(
            r"""
            # s.t. ProductionRateConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the total production quantity is equal to the production hours multiplied by the production rate
            """,
            exercise=exercise,
        )

        exercise_name = "Production and Production Hours"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that the total production quantity is equal to the production hours multiplied by the production rate.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=production_rate,
                needs=[
                    "Production[p,l,t]",
                    "=",
                    "sum{r in RESOURCES}",
                    "ProductionHours[p,l,r,t]",
                    "*",
                    "ProductionRate[p,l,r]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + production_rate)
        else:
            self.add(header + production_rate_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_resource_capacity_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            param AvailableCapacity{r in RESOURCES, l in LOCATIONS} >= 0 default 0; 
                # Available capacity for each resource at each location
            """
        )

        resource_capacity = self._transform(
            r"""
            !exercise!
            s.t. ProductionCapacity{r in RESOURCES, l in LOCATIONS, t in PERIODS}:
                sum{(p, l) in PRODUCTS_LOCATIONS} ProductionHours[p,l,r,t] <= AvailableCapacity[r,l];
                # Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location
            """,
            exercise=exercise,
        )

        resource_capacity_placeholder = self._transform(
            r"""
            # s.t. ProductionCapacity{r in RESOURCES, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location
            """,
            exercise=exercise,
        )

        exercise_name = "Resource capacity"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that the total hours used by all products do not exceed the available capacity for a given resource at each location.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=resource_capacity,
                needs=[
                    "sum{(p, l) in PRODUCTS_LOCATIONS}",
                    "ProductionHours[p,l,r,t]",
                    "<=",
                    "AvailableCapacity[r,l]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + resource_capacity)
        else:
            self.add(header + resource_capacity_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_material_balance_with_transfers_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            set TRANSFER_LANES within {PRODUCTS, LOCATIONS, LOCATIONS};
                # Valid transfer lanes (From_Location, To_Location)
            var Transfers{(p, i, j) in TRANSFER_LANES, t in PERIODS} >= 0;
                # Transfers of product 'p' leaving from location 'i' to location 'j'
            var TransfersIN{(p, l) in PRODUCTS_LOCATIONS, t in PERIODS} = sum {(p, i, l) in TRANSFER_LANES} Transfers[p, i, l, t];
                # Total amount of transfers of product 'p' into location 'l' at period 't'
            var TransfersOUT{(p, l) in PRODUCTS_LOCATIONS, t in PERIODS} = sum {(p, l, i) in TRANSFER_LANES} Transfers[p, l, i, t];
                # Total amount of transfers of product 'p' out of location 'l' at period 't'
            """
        )

        material_balance_with_transfers = self._transform(
            r"""
            !exercise!
            s.t. MaterialBalanceWithTransfers{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                StartingInventory[p,l,t] - MetDemand[p,l,t] + Production[p,l,t]
                + TransfersIN[p,l,t] - TransfersOUT[p,l,t]
                = EndingInventory[p,l,t];
                # Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment
            """,
            exercise=exercise,
        )

        material_balance_with_transfers_placeholder = self._transform(
            r"""
            # s.t. MaterialBalanceWithTransfers{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment
            """,
            exercise=exercise,
        )

        exercise_name = "Material Balance with Transfers"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure material balance by accounting for starting inventory, production, transfers in and out, and demand fulfillment.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=material_balance_with_transfers,
                needs=[
                    "StartingInventory[p,l,t]",
                    "MetDemand[p,l,t]",
                    "Production[p,l,t]",
                    "TransfersIN[p,l,t]",
                    "TransfersOUT[p,l,t]",
                    "=",
                    "EndingInventory[p,l,t]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + material_balance_with_transfers)
        else:
            self.add(header + material_balance_with_transfers_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_target_stock_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            param TargetStock{p in PRODUCTS, l in LOCATIONS} >= 0 default 0;
                # Target stock level for each product and location (could also depend on the period)
            var AboveTarget{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Amount above target stock
            var BelowTarget{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                # Amount below target stock
            """
        )

        target_stock = self._transform(
            r"""
            !exercise!
            s.t. TargetStockConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                TargetStock[p, l] = EndingInventory[p, l, t] + BelowTarget[p, l, t] - AboveTarget[p, l, t];
                # Ensure that the ending inventory is adjusted to either exceed (AboveTarget) or fall below (BelowTarget) the target stock level
            """,
            exercise=exercise,
        )

        target_stock_placeholder = self._transform(
            r"""
            # s.t. TargetStockConstraint{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """,
            exercise=exercise,
        )

        exercise_name = "Target Stocks"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that the ending inventory is adjusted to either exceed (AboveTarget) or fall below (BelowTarget) the target stock level.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=target_stock,
                needs=[
                    "TargetStock[p, l]",
                    "=",
                    "EndingInventory[p, l, t]",
                    "+",
                    "BelowTarget[p, l, t]",
                    "-",
                    "AboveTarget[p, l, t]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + target_stock)
        else:
            self.add(header + target_stock_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_storage_capacity_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            param MaxCapacity{l in LOCATIONS} >= 0;
                # Maximum storage capacity for each location (could also depend on the period)
            """
        )

        storage_capacity = self._transform(
            r"""
            !exercise!
            s.t. StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
                sum{(p, l) in PRODUCTS_LOCATIONS} EndingInventory[p, l, t] <= MaxCapacity[l];
                # Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """,
            exercise=exercise,
        )

        storage_capacity_placeholder = self._transform(
            r"""
            # s.t. StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """,
            exercise=exercise,
        )

        exercise_name = "Storage Capacity"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=storage_capacity,
                needs=[
                    "sum{(p, l) in PRODUCTS_LOCATIONS}",
                    "EndingInventory[p, l, t]",
                    "<=",
                    "MaxCapacity[l]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + storage_capacity)
        else:
            self.add(header + storage_capacity_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_soft_storage_capacity_declaration(self, exercise=None, show=None):
        header = self._transform(
            r"""
            param MaxCapacity{l in LOCATIONS} >= 0;
                # Maximum storage capacity for each location (could also depend on the period)
            var AboveCapacitySlack{l in LOCATIONS, t in PERIODS} >= 0 <= 20;
                # Excess amount of inventory at each location and time period 
            """
        )

        layered_storage_capacity = self._transform(
            r"""
            !exercise!
            s.t. StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
                sum{(p, l) in PRODUCTS_LOCATIONS} EndingInventory[p, l, t] <= MaxCapacity[l] + AboveCapacitySlack[l, t];
                # Ensure that the total ending inventory across all products does not exceed the maximum storage capacity at each location
            """,
            exercise=exercise,
        )

        layered_storage_capacity_placeholder = self._transform(
            r"""
            # s.t. StorageCapacityConstraint{l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the total ending inventory across all products is penalized if it exceeds the maximum storage capacity at each location
            """,
            exercise=exercise,
        )

        exercise_name = "Layered Storage Capacity"

        def render_exercise(ampl, name, number, selected_exercise):
            self._exercise(
                ampl,
                name=name,
                description="Ensure that the total ending inventory across all products is penalized if it exceeds the maximum storage capacity at each location.",
                exercise=number,
                selected_exercise=selected_exercise,
                constraint=layered_storage_capacity,
                needs=[
                    "sum{(p, l) in PRODUCTS_LOCATIONS}",
                    "EndingInventory[p, l, t]",
                    "<=",
                    "MaxCapacity[l]",
                    "AboveCapacitySlack[l, t]",
                ],
            )

        if show or self.show_complete_model:
            self.add(header + layered_storage_capacity)
        else:
            self.add(header + layered_storage_capacity_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_lot_sizing_min(self, use_mp=False, exercise=None, show=None):
        if use_mp:
            header = self._transform(
                r"""
                param MinLotSize default 10;
                """
            )

            lot_sizing = self._transform(
                r"""
                !exercise!
                s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                    Production[p, l, t] = 0 or Production[p, l, t] >= MinLotSize;
                    # Ensure that the production is either 0 or above MinLotSize
                """,
                exercise=exercise,
            )

            exercise_name = "Min Lot-Sizing with High-Level Logic Modeling"

            def render_exercise(ampl, name, number, selected_exercise):
                self._exercise(
                    ampl,
                    name=name,
                    description="Ensure that the production is either 0 or above MinLotSize.",
                    exercise=number,
                    selected_exercise=selected_exercise,
                    constraint=lot_sizing,
                    needs=[
                        "Production[p, l, t]",
                        "=",
                        "0",
                        "or",
                        "Production[p, l, t]",
                        ">=",
                        "MinLotSize",
                    ],
                )

        else:
            header = self._transform(
                r"""
                param MinLotSize default 10;
                param MaxProduction := sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS} Demand[p, l, t];
                !empty!
                var AboveMinLotSize{p in PRODUCTS, l in LOCATIONS, t in PERIODS} >= 0;
                    # Production volume for each product at each location during each time period above the minimum lot size
                var Produce{p in PRODUCTS, l in LOCATIONS, t in PERIODS} binary;
                    # Whether or not we produce each product at each location during each time period
                """
            )

            lot_sizing = self._transform(
                r"""
                !exercise!
                s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                    Production[p, l, t] = Produce[p, l, t] * MinLotSize + AboveMinLotSize[p, l, t] and
                    Production[p, l, t] <= Produce[p, l, t] * MaxProduction and
                    Production[p, l, t] >= Produce[p, l, t] * MinLotSize;
                    # Ensure that the production is either 0 or above MinLotSize
                """,
                exercise=exercise,
            )

            exercise_name = "Min Lot-Sizing with Big-M"

            def render_exercise(ampl, name, number, selected_exercise):
                self._exercise(
                    ampl,
                    name=name,
                    description="Ensure that the production is either 0 or above MinLotSize.",
                    exercise=number,
                    selected_exercise=selected_exercise,
                    constraint=lot_sizing,
                    needs=[
                        "Production[p, l, t]",
                        "Produce[p, l, t]",
                        "MinLotSize",
                        "AboveMinLotSize[p, l, t]",
                        "and",
                        "=",
                        "<=",
                        ">=",
                        "MaxProduction",
                        "MinLotSize",
                    ],
                )

        lot_sizing_placeholder = self._transform(
            r"""
            # s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the production is either 0 or above MinLotSize
            """,
            exercise=exercise,
        )

        if show or self.show_complete_model:
            self.add(header + lot_sizing)
        else:
            self.add(header + lot_sizing_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_lot_sizing_incremental(self, use_mp=False, exercise=None, show=None):
        if use_mp:
            header = self._transform(
                r"""
                param MinLotSize default 10;
                param IncrementLotSize default 5;
                !empty!
                var LSIncrements{p in PRODUCTS, l in LOCATIONS, t in PERIODS} integer >= 0;
                    # Number of lot sizing increments for each product at each location during each time period above the minimum lot size
                """
            )

            lot_sizing = self._transform(
                r"""
                !exercise!
                s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                    Production[p, l, t] = 0 or
                    Production[p, l, t] >= MinLotSize + IncrementLotSize * LSIncrements[p, l, t];
                    # Ensure that the production is either 0 or above MinLotSize + IncrementLotSize * Integer
                """,
                exercise=exercise,
            )

            exercise_name = "Min+Incremental Lot-Sizing with High-Level Logic Modeling"

            def render_exercise(ampl, name, number, selected_exercise):
                self._exercise(
                    ampl,
                    name=name,
                    description="Ensure that the production is either 0 or above MinLotSize + IncrementLotSize * Integer.",
                    exercise=number,
                    selected_exercise=selected_exercise,
                    constraint=lot_sizing,
                    needs=[
                        "Production[p, l, t]",
                        "=",
                        "0",
                        "or",
                        ">=",
                        "MinLotSize",
                        "+",
                        "IncrementLotSize",
                        "*",
                        "LSIncrements[p, l, t]",
                    ],
                )

        else:
            header = self._transform(
                r"""
                param MinLotSize default 10;
                param IncrementLotSize default 5;
                param MaxProduction := sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS} Demand[p, l, t];
                !empty!
                var Produce{p in PRODUCTS, l in LOCATIONS, t in PERIODS} binary;
                    # Whether or not we produce each product at each location during each time period
                var LSIncrements{p in PRODUCTS, l in LOCATIONS, t in PERIODS} integer >= 0;
                    # Number of lot sizing increments for each product at each location during each time period above the minimum lot size
                """
            )

            lot_sizing = self._transform(
                r"""
                !exercise!
                s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
                    Production[p, l, t] = Produce[p, l, t] * MinLotSize + IncrementLotSize * LSIncrements[p, l, t] and
                    Production[p, l, t] <= Produce[p, l, t] * MaxProduction and
                    Production[p, l, t] >= Produce[p, l, t] * MinLotSize;
                    # Ensure that the production is either 0 or above MinLotSize + IncrementLotSize * Integer
                """,
                exercise=exercise,
            )

            exercise_name = "Min+Incremental Lot-Sizing with Big-M"

            def render_exercise(ampl, name, number, selected_exercise):
                self._exercise(
                    ampl,
                    name=name,
                    description="Ensure that the production is either 0 or above MinLotSize + IncrementLotSize * Integer.",
                    exercise=number,
                    selected_exercise=selected_exercise,
                    constraint=lot_sizing,
                    needs=[
                        "Production[p, l, t]",
                        "=",
                        "Produce[p, l, t]",
                        "*",
                        "MinLotSize",
                        "+",
                        "IncrementLotSize",
                        "LSIncrements[p, l, t]",
                        "and",
                        "<=",
                        "MaxProduction",
                        ">=",
                        "MinLotSize",
                    ],
                )

        lot_sizing_placeholder = self._transform(
            r"""
            # s.t. LotSizing{p in PRODUCTS, l in LOCATIONS, t in PERIODS}:
            # ... !exercise!Ensure that the production is either 0 or above MinLotSize + IncrementLotSize * Integer
            """,
            exercise=exercise,
        )

        if show or self.show_complete_model:
            self.add(header + lot_sizing)
        else:
            self.add(header + lot_sizing_placeholder)
            self.exercises.append(
                Exercise(
                    number=exercise,
                    name=exercise_name,
                    render_exercise=render_exercise,
                )
            )

    def add_class1_objective(self):
        self.add(
            r"""
            param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
            param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)

            minimize TotalCost:
                sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS}
                    (UnmetDemandPenalty * UnmetDemand[p, l, t] + EndingInventoryPenalty * EndingInventory[p, l, t]);
                # Objective function to minimize total costs associated with unmet demand and leftover inventory
            """,
            transform=True,
        )

    def add_class1_objective_with_shelf_life(self):
        self.add(
            r"""
            param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
            param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)
            param LostInventoryPenalty default 5;
                # Penalty cost per unit for lost inventory (reflects waste cost)

            minimize TotalCost:
                sum {p in PRODUCTS, l in LOCATIONS, t in PERIODS}
                    (UnmetDemandPenalty * UnmetDemand[p, l, t] + EndingInventoryPenalty * EndingInventory[p, l, t] + LostInventoryPenalty * LostInventory[p, l, t]);
                # Objective function to minimize total costs associated with unmet demand, leftover inventory, and lost inventory
            """,
            transform=True,
        )

    def add_class3_objective(self, layered_storage_capacity, layered_targets):
        parameters = r"""
            param BelowTargetPenalty default 10;
                # Penalty for having inventory below target
            param UnmetDemandPenalty default 10;
                # Penalty cost per unit for unmet demand (impacts decision to meet demand)
            param AboveTargetPenalty default 2;
                # Penalty for having inventory above target
            param EndingInventoryPenalty default 5;
                # Penalty cost per unit for ending inventory (reflects carrying cost)
            param TransferPenalty default 1;
                # Penalty for each unit transferred
            !empty!
            """

        layered_storage_component = ""
        if layered_storage_capacity:
            layered_storage_component = r"""
                + sum {l in LOCATIONS, t in PERIODS} (
                    if AboveCapacitySlack[l, t] > 0 then 
                        (if AboveCapacitySlack[l, t] <= 5 then 10
                                                          else 50)
                )"""

        linear_penalties_objective = (
            parameters
            + r"""
            # Minimize total cost objective
            minimize TotalCost:
                sum{p in PRODUCTS, l in LOCATIONS, t in PERIODS} (
                    UnmetDemandPenalty * UnmetDemand[p, l, t] 
                    + EndingInventoryPenalty * EndingInventory[p, l, t] 
                    + AboveTargetPenalty * AboveTarget[p, l, t] 
                    + BelowTargetPenalty * BelowTarget[p, l, t]
                    + TransferPenalty * TransfersOUT[p, l, t]
                )"""
            + layered_storage_component
            + r""";
                # Objective: Minimize total cost, which includes penalties for unmet demand, ending inventory, deviations from target stock, and transfers
            """
        )

        layered_penalties_objective = (
            parameters
            + r"""
            # Minimize total cost objective
            minimize TotalCost:
                sum{p in PRODUCTS, l in LOCATIONS, t in PERIODS} (
                    UnmetDemandPenalty * UnmetDemand[p, l, t] 
                    + EndingInventoryPenalty * EndingInventory[p, l, t]
                    + (if AboveTarget[p, l, t] <= 5 then AboveTarget[p, l, t] * AboveTargetPenalty
                                                    else 10 * AboveTargetPenalty)
                    + (if BelowTarget[p, l, t] <= 5 then BelowTarget[p, l, t] * BelowTargetPenalty
                                                    else 10 * BelowTargetPenalty)
                    + TransferPenalty * TransfersOUT[p, l, t]
                )"""
            + layered_storage_component
            + r""";
                # Objective: Minimize total cost, which includes penalties for unmet demand, ending inventory, deviations from target stock, and transfers
            """
        )
        if layered_targets:
            self.add(layered_penalties_objective, transform=True)
        else:
            self.add(linear_penalties_objective, transform=True)

    def display_exercises(self, ampl):
        if self.exercises == []:
            return
        st.markdown("## 🧑‍🏫 Exercises")

        exercises_lst = ["", "All"] + [e.title() for e in self.exercises]
        selected_exercise = (
            exercises_lst.index(
                st.selectbox(
                    "Select the exercise(s) you want to complete 👇",
                    exercises_lst,
                    key="exercise",
                    index=0,
                    on_change=self.on_change,
                )
            )
            - 1
        )
        for e in self.exercises:
            e.render(ampl=ampl, selected_exercise=selected_exercise)

    def adjust_parameters(self, ampl):
        parameter_controllers = {
            "UnmetDemandPenalty": lambda: st.slider(
                "UnmetDemandPenalty:",
                min_value=0,
                max_value=50,
                value=10,
                on_change=self.on_change,
            ),
            "MaxShelfLife": lambda: st.slider(
                "MaxShelfLife:",
                min_value=0,
                max_value=5,
                value=3,
                on_change=self.on_change,
            ),
            "EnsureOldStockGoesFirst": lambda: st.checkbox(
                "Sell old inventory first", value=True
            ),
            "EndingInventoryPenalty": lambda: st.slider(
                "EndingInventoryPenalty:",
                min_value=0,
                max_value=50,
                value=5,
                on_change=self.on_change,
            ),
            "LostInventoryPenalty": lambda: st.slider(
                "LostInventoryPenalty:",
                min_value=0,
                max_value=50,
                value=10,
                on_change=self.on_change,
            ),
            "AboveTargetPenalty": lambda: st.slider(
                "AboveTargetPenalty:",
                min_value=0,
                max_value=50,
                value=2,
                on_change=self.on_change,
            ),
            "BelowTargetPenalty": lambda: st.slider(
                "BelowTargetPenalty:",
                min_value=0,
                max_value=50,
                value=10,
                on_change=self.on_change,
            ),
            "TransferPenalty": lambda: st.slider(
                "TransferPenalty:",
                min_value=0,
                max_value=50,
                value=1,
                on_change=self.on_change,
            ),
            "MinLotSize": lambda: st.slider(
                "MinLotSize:",
                min_value=0,
                max_value=50,
                value=10,
                on_change=self.on_change,
            ),
            "IncrementLotSize": lambda: st.slider(
                "IncrementLotSize:",
                min_value=0,
                max_value=50,
                value=5,
                on_change=self.on_change,
            ),
        }

        p = 0
        with st.expander("Adjust parameters"):
            cols = st.columns(2)
            model_parameters = set(ampl.get_data("_PARS").to_list())  # FIXME
            for parameter, controller in parameter_controllers.items():
                if parameter not in model_parameters:
                    continue
                with cols[p % 2]:
                    ampl.param[parameter] = controller()
                    p += 1
