Kondili_H = 10
Kondili_STN = {
    # time grid
    "TIME": list(range(0, Kondili_H + 1)),
    # states
    "STATES": {
        "Feed_A": {"capacity": 500, "initial": 500, "price": 0},
        "Feed_B": {"capacity": 500, "initial": 500, "price": 0},
        "Feed_C": {"capacity": 500, "initial": 500, "price": 0},
        "Hot_A": {"capacity": 100, "initial": 0, "price": -100},
        "Int_AB": {"capacity": 200, "initial": 0, "price": -100},
        "Int_BC": {"capacity": 150, "initial": 0, "price": -100},
        "Impure_E": {"capacity": 100, "initial": 0, "price": -100},
        "Product_1": {"capacity": 500, "initial": 0, "price": 10},
        "Product_2": {"capacity": 500, "initial": 0, "price": 10},
    },
    # state-to-task arcs indexed by (state, task)
    "ST_ARCS": {
        ("Feed_A", "Heating"): {"rho": 1.0},
        ("Feed_B", "Reaction_1"): {"rho": 0.5},
        ("Feed_C", "Reaction_1"): {"rho": 0.5},
        ("Feed_C", "Reaction_3"): {"rho": 0.2},
        ("Hot_A", "Reaction_2"): {"rho": 0.4},
        ("Int_AB", "Reaction_3"): {"rho": 0.8},
        ("Int_BC", "Reaction_2"): {"rho": 0.6},
        ("Impure_E", "Separation"): {"rho": 1.0},
    },
    # task-to-state arcs indexed by (task, state)
    "TS_ARCS": {
        ("Heating", "Hot_A"): {"dur": 1, "rho": 1.0},
        ("Reaction_2", "Product_1"): {"dur": 2, "rho": 0.4},
        ("Reaction_2", "Int_AB"): {"dur": 2, "rho": 0.6},
        ("Reaction_1", "Int_BC"): {"dur": 2, "rho": 1.0},
        ("Reaction_3", "Impure_E"): {"dur": 1, "rho": 1.0},
        ("Separation", "Int_AB"): {"dur": 2, "rho": 0.1},
        ("Separation", "Product_2"): {"dur": 1, "rho": 0.9},
    },
    # unit data indexed by (unit, task)
    "UNIT_TASKS": {
        ("Heater", "Heating"): {
            "Bmin": 0,
            "Bmax": 100,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_1"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_2"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_3"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_1"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_2"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_3"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Still", "Separation"): {
            "Bmin": 0,
            "Bmax": 200,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
    },
}

Hydrolubes_H = 16
Hydrolubes_STN = {
    # time grid
    "TIME": list(range(0, Hydrolubes_H + 1)),
    # states
    "STATES": {
        "Feed_A": {"capacity": 500, "initial": 500, "price": 0},
        "Feed_B": {"capacity": 500, "initial": 500, "price": 0},
        "Feed_C": {"capacity": 500, "initial": 500, "price": 0},
        "Hot_A": {"capacity": 100, "initial": 0, "price": -100},
        "Int_AB": {"capacity": 200, "initial": 0, "price": -100},
        "Int_BC": {"capacity": 150, "initial": 0, "price": -100},
        "Impure_E": {"capacity": 100, "initial": 0, "price": -100},
        "Product_1": {"capacity": 500, "initial": 0, "price": 10},
        "Product_2": {"capacity": 500, "initial": 0, "price": 10},
    },
    # state-to-task arcs indexed by (state, task)
    "ST_ARCS": {
        ("Feed_A", "Heating"): {"rho": 1.0},
        ("Feed_B", "Reaction_1"): {"rho": 0.5},
        ("Feed_C", "Reaction_1"): {"rho": 0.5},
        ("Feed_C", "Reaction_3"): {"rho": 0.2},
        ("Hot_A", "Reaction_2"): {"rho": 0.4},
        ("Int_AB", "Reaction_3"): {"rho": 0.8},
        ("Int_BC", "Reaction_2"): {"rho": 0.6},
        ("Impure_E", "Separation"): {"rho": 1.0},
    },
    # task-to-state arcs indexed by (task, state)
    "TS_ARCS": {
        ("Heating", "Hot_A"): {"dur": 1, "rho": 1.0},
        ("Reaction_2", "Product_1"): {"dur": 2, "rho": 0.4},
        ("Reaction_2", "Int_AB"): {"dur": 2, "rho": 0.6},
        ("Reaction_1", "Int_BC"): {"dur": 2, "rho": 1.0},
        ("Reaction_3", "Impure_E"): {"dur": 1, "rho": 1.0},
        ("Separation", "Int_AB"): {"dur": 2, "rho": 0.1},
        ("Separation", "Product_2"): {"dur": 1, "rho": 0.9},
    },
    # unit data indexed by (unit, task)
    "UNIT_TASKS": {
        ("Heater", "Heating"): {
            "Bmin": 0,
            "Bmax": 100,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_1"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_2"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_1", "Reaction_3"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_1"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_2"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Reactor_2", "Reaction_3"): {
            "Bmin": 0,
            "Bmax": 80,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
        ("Still", "Separation"): {
            "Bmin": 0,
            "Bmax": 200,
            "Cost": 1,
            "vCost": 0,
            "Tclean": 0,
        },
    },
}
