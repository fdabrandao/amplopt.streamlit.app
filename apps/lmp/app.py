import streamlit as st
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt
import networkx as nx
from ..common import solver_selector, MP_SOLVERS_LINKS


LMP_MODEL = r"""
# Nodes and transmission lines (as node pairs)
set NODES;
set LINES within {NODES, NODES};

# Parameters: generation cost, capacity, demand, and line limits
param generation_cost{NODES};
param generation_capacity{NODES};
param demand{NODES};
param line_capacity{LINES};

# Variables: generation at nodes and flow on lines
var Generation{i in NODES} >= 0, <= generation_capacity[i];
var Flow{LINES};

# Objective: minimize total generation cost
minimize TotalCost:
    sum {n in NODES} generation_cost[n] * Generation[n];

# Power balance at each node: generation + inflow - outflow = demand
s.t. Balance {n in NODES}:
    Generation[n] + sum {(i,n) in LINES} Flow[i,n] - sum {(n,j) in LINES} Flow[n,j] = demand[n];

# Flow limits on each line
s.t. LineLimits {(i,j) in LINES}:
    -line_capacity[i,j] <= Flow[i,j] <= line_capacity[i,j];
"""


def main():
    st.title("⚡ Locational Marginal Pricing")

    st.markdown(
        """
        **What is Location Marginal Pricing?**

        Location Marginal Pricing (LMP) is a method used in electricity markets to determine the price of electricity at different locations, or nodes, in the power grid. LMP reflects the marginal cost of supplying the next increment of electricity demand at a specific location, accounting for generation costs, demand, and the physical limitations of the transmission system. 

        This pricing mechanism ensures that prices signal both energy value and congestion, promoting efficient use of generation and transmission resources.
        
        Learn more on [Optimization Models in Electricity Markets](https://dev.ampl.com/ampl/books/anthony-papavasiliou/electricity-markets.html) by Anthony Papavasiliou. It is is a textbook published by Cambridge University Press that treats the analysis of optimization models that are routinely used in electricity market operations.
        """
    )

    nodes = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Node 6", "Node 7"]
    lines = [
        ("Node 1", "Node 2"),
        ("Node 2", "Node 3"),
        ("Node 3", "Node 4"),
        ("Node 4", "Node 5"),
        ("Node 5", "Node 3"),
        ("Node 5", "Node 6"),
        ("Node 6", "Node 7"),
        ("Node 7", "Node 5"),
    ]

    # User Inputs
    st.sidebar.header("Generator Parameters")
    gen_costs = {
        n: st.sidebar.slider(f"Gen Cost {n} ($/MWh)", 10, 100, 20 + i * 10)
        for i, n in enumerate(nodes)
    }
    gen_caps = {
        n: st.sidebar.slider(f"Gen Cap {n} (MW)", 0, 100, 50 + i * 10)
        for i, n in enumerate(nodes)
    }

    demands = {
        n: st.sidebar.slider(f"Demand {n} (MW)", 0, 100, 30 + i * 5)
        for i, n in enumerate(nodes)
    }

    st.sidebar.header("Transmission Line Capacities")
    line_caps = {
        f"{i}-{j}": st.sidebar.slider(f"Cap {i}-{j} (MW)", 0, 100, 40) for i, j in lines
    }

    st.write("## Simplified Electricity Market Simulation")
    st.code(LMP_MODEL)

    # Load model in amplpy
    ampl = AMPL()
    ampl.eval(LMP_MODEL)
    ampl.set["NODES"] = nodes
    ampl.set["LINES"] = lines
    ampl.param["generation_cost"] = gen_costs
    ampl.param["generation_capacity"] = gen_caps
    ampl.param["demand"] = demands
    ampl.param["line_capacity"] = {(i, j): line_caps[f"{i}-{j}"] for i, j in lines}

    # Select the solver to use
    solver, _ = solver_selector(mp_only=True)
    # Solve the problem
    output = ampl.solve(
        solver=solver,
        mp_options="outlev=1",
        return_output=True,
    )

    if ampl.solve_result not in ["solved", "limit"]:
        st.error(f"The model could not be solved:\n```\n{output}\n```")
    else:
        with st.expander("Solver output"):
            st.write(f"```\n{output}\n```")

        # Retrieve results
        gen_df = ampl.var["Generation"].to_pandas()
        lmp_values = ampl.get_data("Balance.dual").to_pandas()

        lmp_df = pd.DataFrame(
            {
                "Node": gen_df.index,
                "Generation (MW)": gen_df.iloc[:, 0],
                "LMP ($/MWh)": lmp_values.iloc[:, 0],
            }
        ).set_index("Node")

        st.markdown("## Results")
        # st.dataframe(lmp_df)
        # st.bar_chart(lmp_df[["LMP ($/MWh)"]], y_label="LMP ($/MWh)")
        # st.bar_chart(lmp_df[["Generation (MW)"]], y_label="Generation (MW)")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart with LMP values
            st.markdown("### LMP ($/MWh)")
            fig, ax = plt.subplots()
            ax.bar(lmp_df.index, lmp_df["LMP ($/MWh)"])
            for i, val in enumerate(lmp_df["LMP ($/MWh)"]):
                ax.text(i, val + 0.5, f"{val:.2f}", ha="center", va="bottom")
            ax.set_ylabel("LMP ($/MWh)")
            ax.set_title("Location Marginal Prices by Node")
            st.pyplot(fig)

        with col2:
            # Bar chart with Generation values
            st.markdown("### Generation (MW)")
            fig2, ax2 = plt.subplots()
            ax2.bar(lmp_df.index, lmp_df["Generation (MW)"])
            for i, val in enumerate(lmp_df["Generation (MW)"]):
                ax2.text(i, val + 0.5, f"{val:.2f}", ha="center", va="bottom")
            ax2.set_ylabel("Generation (MW)")
            ax2.set_title("Generation Output by Node")
            st.pyplot(fig2)

        # Visualize Network
        st.markdown("## Network Visualization")
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(lines)
        pos = nx.spring_layout(G, seed=42)

        edge_labels = {
            (i, j) if flow > 0 else (j, i): f"{abs(flow)} MW / {capacity} MW"
            for (i, j), (flow, capacity) in ampl.get_data("Flow", "line_capacity")
            .to_dict()
            .items()
        }
        node_labels = {
            i: f"\n{i}\n{lmp} $/MWh\n{g} / {glimit} MW\n-{d} MW\n"
            for i, (lmp, d, g, glimit) in ampl.get_data(
                "Balance.dual", "demand", "Generation", "generation_capacity"
            )
            .to_dict()
            .items()
        }

        plt.figure(figsize=(8, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="orange",
            node_size=4000,
            font_size=10,
            labels=node_labels,
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        st.pyplot(plt)

    st.markdown(
        """
    #### [[Source Code on GitHub](http://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/lmp)]
    """
    )
