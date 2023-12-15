import streamlit as st
from amplpy import AMPL, modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    st.markdown(
        """
    # 🌎 Global Optimization Has Arrived!
    
    Christmas 🎄 Problem: Optimize the placement of ornaments on a tree so that they are equidistant!

    ```python
    # Define parameters
    param n;           # Number of ornaments
    param width;       # Maximum width for placement of ornaments
    param sine_slope;  # Slope for the sine functions
    param tree_slope;  # Slope for the tree shape
    param offset;      # Offset for the sine function
    param frequency;   # Frequency for the sine function

    # Define a set for the ornaments
    set ORNAMENTS ordered := 1..n;  # Ordered set representing the ornaments

    # Variables
    var X{ORNAMENTS} >= 0 <= width;  # X-coordinate of each ornament within the specified width
    var Y{i in ORNAMENTS} = sin(frequency * X[i]) + offset + sine_slope * X[i];  # Y-coordinate using a sine function

    # Objective function
    maximize MinDistance:  # Objective: Maximize the minimum distance between consecutive ornaments
        min{i in ORNAMENTS: ord(i) > 1} sqrt((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

    # Constraints
    s.t. Order{i in ORNAMENTS: ord(i) > 1}:  # Ensure the ornaments are ordered from left to right
        X[i] >= X[i-1];

    s.t. TreeShape{i in ORNAMENTS}:  # Constraints for the shape of the tree
        Y[i] <= min(tree_slope * X[i], tree_slope * (width - X[i]));
    ```
    """
    )

    ampl = AMPL()
    ampl.eval(
        r"""
    # Define parameters
    param n;           # Number of ornaments
    param width;       # Maximum width for placement of ornaments
    param sine_slope;  # Slope for the sine functions
    param tree_slope;  # Slope for the tree shape
    param offset;      # Offset for the sine function
    param frequency;   # Frequency for the sine function

    # Define a set for the ornaments
    set ORNAMENTS ordered := 1..n;  # Ordered set representing the ornaments

    # Variables
    var X{ORNAMENTS} >= 0 <= width;  # X-coordinate of each ornament within the specified width
    var Y{i in ORNAMENTS} = sin(frequency * X[i]) + offset + sine_slope * X[i];  # Y-coordinate using a sine function

    # Objective function
    maximize MinDistance:  # Objective: Maximize the minimum distance between consecutive ornaments
        min{i in ORNAMENTS: ord(i) > 1} sqrt((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

    # Constraints
    s.t. Order{i in ORNAMENTS: ord(i) > 1}:  # Ensure the ornaments are ordered from left to right
        X[i] >= X[i-1];

    s.t. TreeShape{i in ORNAMENTS}:  # Constraints for the shape of the tree
        Y[i] <= min(tree_slope * X[i], tree_slope * (width - X[i]));

    # s.t. PositionFirstOrnament:  # Position the first ornament at the correct Y-coordinate based on the slope
    #     Y[first(ORNAMENTS)] = tree_slope * X[first(ORNAMENTS)];

    # s.t. PositionLastOrnament:  # Position the last ornament at the correct Y-coordinate based on the slope and width
    #     Y[last(ORNAMENTS)] = tree_slope * (width - X[last(ORNAMENTS)]);
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        width = st.slider(
            "Tree width? 👇", 3 * math.pi, 6 * math.pi, 4 * math.pi, step=math.pi
        )
        tree_slope = st.slider("Tree slope? 👇", 1, 5, 5)
        sine_slope = st.slider("Slope for the ornaments? 👇", 0.0, 1.0, 0.7, step=0.1)
        frequency = st.slider(
            "Frequency value for `sin(frequency * x)`? 👇", 1.0, 3.0, 1.0, step=0.5
        )
        per_cycle = st.slider("How many ornaments per cycle? 👇", 1, 3, 2)
        height = width / 2.0 * tree_slope
        nlevels = st.slider("How many levels? 👇", 2, 10, 5)
        solvers = [
            "Gurobi 🚀 (Global)",
            "SCIP (Global)",
            "LGO (Global)",
            # "LindoGlobal (Global)",
            "Octeract (Global)",
            "Knitro",
            # "CONOPT",
            # "LOQO",
            # "MINOS",
            # "SNOPT",
            # "BARON",
            # "IPOPT",
            # "Couenne",
            # "Bonmin",
        ]
        solver = st.selectbox("Pick the solver 👇", solvers, key="solver")
        if " " in solver:
            solver = solver[: solver.find(" ")]
        solver = solver.lower()

    ampl.param["width"] = width
    ampl.param["tree_slope"] = tree_slope
    ampl.param["sine_slope"] = sine_slope
    ampl.param["frequency"] = frequency
    ampl.option["solver"] = solver
    ampl.option["gurobi_options"] = "global=1 timelim=10 outlev=1"
    ampl.option["scip_options"] = "timelim=10 outlev=1"
    ampl.option["lindoglobal_options"] = "maxtime=10"
    ampl.option["knitro_options"] = "maxtime_cpu=10"
    if ampl.option[f"{solver}_options"] == "":
        ampl.option[f"{solver}_options"] = "timelim=10"

    def solve(n, offset):
        ampl.param["n"] = n
        ampl.param["offset"] = offset
        solve_output = ampl.get_output("solve;")
        # assert ampl.solve_result == "solved"
        return ampl.get_data("X, Y").to_pandas(), {
            "solve_result": ampl.solve_result,
            "solve_time": ampl.get_value("_solve_time"),
            "objective_value": ampl.get_value("MinDistance"),
            "solver_output": solve_output,
        }

    plt.figure(figsize=(5, 5), dpi=80)
    x = np.linspace(0, width, 1000)
    tree_left = tree_slope * x
    tree_right = tree_slope * (width - x)

    x_line1 = np.linspace(0, width / 2, 1000)
    plt.plot(x_line1, tree_slope * x_line1, color="green", linestyle="--")
    x_line2 = np.linspace(width / 2, width, 1000)
    plt.plot(x_line2, tree_slope * (width - x_line2), color="green", linestyle="--")

    solve_info = {}
    for i in range(nlevels):
        offset = i * height / float(nlevels)
        color = ["red", "blue"][i % 2]

        sin_line = np.sin(frequency * x) + offset + sine_slope * x
        x_line = x[(sin_line < tree_left) & (sin_line < tree_right)]
        y_line = sin_line[(sin_line < tree_left) & (sin_line < tree_right)]
        plt.plot(x_line, y_line, color=color)

        total_length = np.max(x_line) - np.min(x_line)
        cycles = frequency * total_length / (2 * math.pi)
        n_ornaments = min(max(2, int(round(per_cycle * cycles))), 10)

        solution, solve_info[i + 1] = solve(n=n_ornaments, offset=offset)
        plt.scatter(solution.X, solution.Y, color="green", s=100)

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)

    # Set aspect ratio to 'equal' for a square grid
    plt.gca().set_aspect("equal", adjustable="box")

    with col1:
        st.markdown("Results for each level:")
        st.write(pd.DataFrame.from_dict(solve_info, orient="index"))

    with col2:
        st.pyplot(plt)

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/global_optimization)]
    """
    )
