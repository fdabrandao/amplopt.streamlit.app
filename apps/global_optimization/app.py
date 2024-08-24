import streamlit as st
from amplpy import AMPL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
import random
import math


class ChristmasTreeOptimizer:
    def __init__(
        self, width: float, height: float, sine_slope: float, frequency: float
    ):
        ampl = AMPL()
        ampl.eval(
            r"""
        # Define parameters
        param n;           # Number of ornaments
        param width;       # Tree width
        param height;      # Tree height
        param offset;      # Offset of the sine function
        param frequency;   # Frequency of the sine function
        param sine_slope;  # Slope of the sine functions
        param tree_slope :=  height / (width/2);  # Slope of the tree shape

        # Define a set for the ornaments
        set ORNAMENTS ordered := 1..n;  # Ordered set representing the ornaments

        # Variables
        var X{ORNAMENTS} >= 0 <= width;  # X-coordinate of each ornament within the specified width
        var Y{i in ORNAMENTS} = sin(frequency * X[i]) + sine_slope * X[i] + offset;  # Y-coordinate using a sine function

        # Objective functions
        maximize MinEuclideanDistance:  # Objective: Maximize the minimum euclidean distance between consecutive ornaments
            min{i in ORNAMENTS: ord(i) > 1} sqrt((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

        maximize MinSquaredEuclideanDistance:  # Objective: Maximize the minimum squared euclidean distance between consecutive ornaments
            min{i in ORNAMENTS: ord(i) > 1} ((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

        maximize MinManhattanDistance:  # Objective: Maximize the minimum manhattan distance between consecutive ornaments
            min{i in ORNAMENTS: ord(i) > 1} (abs(X[i] - X[i-1]) + abs(Y[i] - Y[i-1]));

        # Constraints
        s.t. Order{i in ORNAMENTS: ord(i) > 1}:  # Ensure the ornaments are ordered from left to right
            X[i] >= X[i-1];

        s.t. TreeShape{i in ORNAMENTS}:  # Constraints for the shape of the tree
            Y[i] <= min(tree_slope * X[i], tree_slope * (width - X[i]));
        """
        )
        ampl.param["width"] = width
        ampl.param["height"] = height
        ampl.param["sine_slope"] = sine_slope
        ampl.param["frequency"] = frequency
        self.ampl = ampl

    def solve(self, solver: str, objective: str, n: int, offset: float):
        ampl = self.ampl
        ampl.param["n"] = n
        ampl.param["offset"] = offset
        ampl.option["solver"] = solver
        solve_output = ampl.get_output(f"solve {objective};")
        return ampl.get_data("X, Y").to_pandas(), {
            "solve_result": ampl.solve_result,
            "solve_time": ampl.get_value("_solve_elapsed_time"),
            "objective_value": ampl.get_value(objective),
            "solver_output": solve_output,
        }


def decorate_tree(
    optimizer: ChristmasTreeOptimizer,
    solver: str,
    objective: str,
    tree_color: str,
    nlevels: int,
    per_cycle: int,
):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80, facecolor="none")
    ax.set_facecolor("none")
    fig.gca().set_aspect("equal", adjustable="box")
    # ax.axhline(0, color="black", linewidth=0.5)
    # ax.axvline(0, color="black", linewidth=0.5)
    # ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ampl = optimizer.ampl
    width = ampl.get_value("width")
    height = ampl.get_value("height")
    tree_slope = ampl.get_value("tree_slope")
    frequency = ampl.get_value("frequency")
    sine_slope = ampl.get_value("sine_slope")

    x = np.linspace(0, width, 1000)
    tree_left = tree_slope * x
    tree_right = tree_slope * (width - x)

    # Draw the borders of the tree
    x_line1 = np.linspace(0, width / 2, 1000)
    ax.plot(
        x_line1,
        tree_slope * x_line1,
        color=tree_color,
        linestyle="-",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
    )
    x_line2 = np.linspace(width / 2, width, 1000)
    ax.plot(
        x_line2,
        tree_slope * (width - x_line2),
        color=tree_color,
        linestyle="-",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
    )

    ax.text(
        width / 2,
        height,
        "★",
        fontsize=25,
        ha="center",
        va="center",
        color="gold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="orange")],
    )

    # Calculate the minimum values between the two functions
    tree_y1 = tree_slope * x
    tree_y2 = tree_slope * (width - x)
    tree_y_min = np.minimum(tree_y1, tree_y2)
    # Filling the area where values are smaller than both lines with green color
    ax.fill_between(
        x,
        tree_y_min,
        where=(tree_y_min <= tree_y1) & (tree_y_min <= tree_y2),
        color=tree_color,
        alpha=0.3,
    )

    # Plot lines and ornaments
    solve_info = {}
    ornament_colors = ["red", "green", "blue", "orange", "purple", "white"]
    ornament_colors = [color for color in ornament_colors if color != tree_color]
    ornament_colors = random.sample(ornament_colors, 2)
    for i in range(nlevels):
        offset = i * height / float(nlevels + 1)
        color = ornament_colors[i % len(ornament_colors)]

        # Plot lines
        sin_line = np.sin(frequency * x) + sine_slope * x + offset
        x_line = x[(sin_line < tree_left) & (sin_line < tree_right)]
        y_line = sin_line[(sin_line < tree_left) & (sin_line < tree_right)]
        if len(x_line) == 0:
            continue
        ax.plot(
            x_line,
            y_line,
            color=color,
            path_effects=[patheffects.withStroke(linewidth=5, foreground="gold")],
        )

        # Calculate number of ornaments
        total_length = np.max(x_line) - np.min(x_line)
        cycles = frequency * total_length / (2 * math.pi)
        n_ornaments = min(max(3, int(round(per_cycle * cycles))), 10)
        if i == nlevels - 1:
            n_ornaments = 2

        # Solve optimization problem
        solution, solve_info[i + 1] = optimizer.solve(
            solver=solver,
            objective=objective,
            n=n_ornaments,
            offset=offset,
        )
        # Plot ornaments
        ax.scatter(
            solution.X,
            solution.Y,
            color=color,
            edgecolor="gold",
            zorder=3,
            s=100,
        )
    return fig, ax, solve_info


def main():
    st.title("🎅 Global Non-Linear Optimization")
    st.markdown(
        r"""
        Global non-linear optimization involves finding the optimal solution for a
        problem with multiple variables, where the objective function and constraints 
        are non-linear, and the aim is to discover the global maximum or minimum across 
        the entire feasible space. Unlike local optimization, which seeks the best 
        solution within a limited region, global optimization seeks the overall best
        solution within the entire feasible domain, often requiring extensive exploration 
        of the solution space. 

        ## Christmas 🎄 Problem
        
        Optimize the placement of ornaments on a tree 🎄 so that
        we maximize the minimum Euclidean or Manhattan distance between consecutive
        ornaments. The following AMPL model optimizes the placement of ornaments on 
        a sinusoidal line in such a way that we maximize the minimum distance
        between each of them. It can be solved for multiple lines in order to 
        decorate an entire tree.
        """
    )
    social_media = " ".join(
        [
            "[[Colab Notebook](https://colab.research.google.com/github/ampl/colab.ampl.com/blob/master/authors/fdabrandao/global/christmas_tree.ipynb)]",
            "[[LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7143238663445950465)]",
            "[[Twitter Post](https://twitter.com/AMPLopt/status/1737472923349184545)]",
            "[[Youtube Short](https://www.youtube.com/shorts/QWWanzp8c-0)]",
        ]
    )
    st.markdown(social_media)
    st.markdown(
        r"""
    ```python
    # Define parameters
    param n;           # Number of ornaments
    param width;       # Tree width
    param height;      # Tree height
    param offset;      # Offset of the sine function
    param frequency;   # Frequency of the sine function
    param sine_slope;  # Slope of the sine functions
    param tree_slope :=  height / (width/2);  # Slope of the tree shape

    # Define a set for the ornaments
    set ORNAMENTS ordered := 1..n;  # Ordered set representing the ornaments

    # Variables
    var X{ORNAMENTS} >= 0 <= width;  # X-coordinate of each ornament within the specified width
    var Y{i in ORNAMENTS} = sin(frequency * X[i]) + sine_slope * X[i] + offset;  # Y-coordinate using a sine function

    # Objective functions
    maximize MinEuclideanDistance:  # Objective: Maximize the minimum euclidean distance between consecutive ornaments
        min{i in ORNAMENTS: ord(i) > 1} sqrt((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

    maximize MinSquaredEuclideanDistance:  # Objective: Maximize the minimum squared euclidean distance between consecutive ornaments
        min{i in ORNAMENTS: ord(i) > 1} ((X[i] - X[i-1])^2 + (Y[i] - Y[i-1])^2);

    maximize MinManhattanDistance:  # Objective: Maximize the minimum manhattan distance between consecutive ornaments
        min{i in ORNAMENTS: ord(i) > 1} (abs(X[i] - X[i-1]) + abs(Y[i] - Y[i-1]));

    # Constraints
    s.t. Order{i in ORNAMENTS: ord(i) > 1}:  # Ensure the ornaments are ordered from left to right
        X[i] >= X[i-1];

    s.t. TreeShape{i in ORNAMENTS}:  # Constraints for the shape of the tree
        Y[i] <= min(tree_slope * X[i], tree_slope * (width - X[i]));
    ```

    Learn more on our [Colab Notebook](https://colab.research.google.com/github/ampl/colab.ampl.com/blob/master/authors/fdabrandao/global/christmas_tree.ipynb).

    ### Optimize your Christmas 🎄 to Global Optimality!
    """
    )

    left, right = st.columns(2)

    with left:
        height = 20
        c1, c2 = st.columns(2)
        with c1:
            width_height_ration = st.slider(
                "🎄 width/height ratio 👇", 0.35, 0.65, 0.4, step=0.05
            )
            width = height * width_height_ration
        with c2:
            tree_colors = [
                "green",
                "silver",
                "gold",
            ]
            tree_color = st.selectbox("🎄 color 👇", tree_colors, key="tree_color")

        c1, c2 = st.columns(2)
        with c1:
            nlevels = st.slider("Number of waves 👇", 2, 8, 5)
        with c2:
            sine_slope = st.slider("Wave slope 👇", 0.0, 0.9, 0.7, step=0.1)

        c1, c2 = st.columns(2)
        with c1:
            frequency = st.slider("Wave oscillation rate 👇", 1.0, 3.0, 1.0, step=0.5)
        with c2:
            per_cycle = st.slider("Max ornaments per cycle 👇", 1, 3, 2)

        solvers = [
            "Gurobi 11 🚀 (with global=1)",
            "SCIP",
            "LGO",
            "Octeract",
            "Knitro (Local)",
        ]
        solver = st.selectbox("Solver 👇", solvers, key="solver")
        if " " in solver:
            solver = solver[: solver.find(" ")]
        solver = solver.lower()

        objectives = [
            "maximize MinEuclideanDistance",
            "maximize MinSquaredEuclideanDistance",
            "maximize MinManhattanDistance",
        ]
        objective = st.selectbox("Objective 👇", objectives, key="objective")
        objective = objective[objective.find(" ") + 1 :]

    # Create ChristmasTreeOptimizer object to optimize the placement of the ornaments
    optimizer = ChristmasTreeOptimizer(width, height, sine_slope, frequency)

    # Set solver options such as timelim
    optimizer.ampl.option["gurobi_options"] = "global=1 timelim=5 outlev=1"
    optimizer.ampl.option["scip_options"] = "timelim=5 outlev=1"
    optimizer.ampl.option["lindoglobal_options"] = "maxtime=5"
    optimizer.ampl.option["knitro_options"] = "maxtime_cpu=5"
    optimizer.ampl.option["octeract_options"] = "MAX_SOLVER_TIME=5"
    if optimizer.ampl.option[f"{solver}_options"] == "":
        optimizer.ampl.option[f"{solver}_options"] = "timelim=5"

    # Optimize tree decoration
    fig, _, solve_info = decorate_tree(
        optimizer, solver, objective, tree_color, nlevels, per_cycle
    )

    with left:
        st.markdown("Solve results for each wave:")
        st.write(pd.DataFrame.from_dict(solve_info, orient="index"))

    with right:
        st.pyplot(fig)

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/global_optimization)] """
        + social_media
    )
