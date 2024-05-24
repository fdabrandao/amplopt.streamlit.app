import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt


def remove_isolated_states(stn, verbose=False):
    states = set()
    for state in stn["STATES"]:
        states.add(state)

    tasks = set()
    connected_tasks = set()
    connected_states = set()
    for s, t in stn["ST_ARCS"]:
        tasks.add(t)
        if s not in states:
            continue
        connected_tasks.add(t)
        connected_states.add(s)
    for t, s in stn["TS_ARCS"]:
        tasks.add(t)
        if s not in states:
            continue
        connected_tasks.add(t)
        connected_states.add(s)
    units = set()
    connected_units = set()
    tasks_with_units = set()
    for u, t in stn["UNIT_TASKS"]:
        units.add(u)
        tasks_with_units.add(t)
        if t not in connected_tasks:
            continue
        connected_units.add(u)

    if verbose:
        print(f"isolated satates: {states-connected_states}")
        print(f"isolated tasks: {tasks-connected_tasks}")
        print(f"isolated units: {units-connected_units}")
        print(f"tasks without units: {tasks-tasks_with_units}")

    return (
        {
            # time grid
            "TIME": stn["TIME"],
            # states
            "STATES": {s: stn["STATES"][s] for s in connected_states},
            # state-to-task arcs indexed by (state, task)
            "ST_ARCS": {
                (s, t): stn["ST_ARCS"][(s, t)]
                for (s, t) in stn["ST_ARCS"]
                if s in connected_states and t in connected_tasks
            },
            # task-to-state arcs indexed by (task, state)
            "TS_ARCS": {
                (t, s): stn["TS_ARCS"][(t, s)]
                for (t, s) in stn["TS_ARCS"]
                if s in connected_states and t in connected_tasks
            },
            # unit data indexed by (unit, task)
            "UNIT_TASKS": {
                (u, t): stn["UNIT_TASKS"][(u, t)]
                for (u, t) in stn["UNIT_TASKS"]
                if t in connected_tasks
            },
        },
        connected_states,
        connected_tasks,
    )


def build_graph(stn, verbose=False):
    # Remove isolated nodes
    stn, connected_states, connected_tasks = remove_isolated_states(
        stn, verbose=verbose
    )
    # Create a directed graph
    graph = nx.DiGraph()

    # Add states and tasks as nodes
    graph.add_nodes_from(connected_states, type="state")
    graph.add_nodes_from(connected_tasks, type="task")

    # Add state-to-task and task-to-state arcs
    for (s, t), v in stn["ST_ARCS"].items():
        if s not in connected_states or t not in connected_tasks:
            continue
        graph.add_edge(s, t, rho=v["rho"])
    for (t, s), v in stn["TS_ARCS"].items():
        if s not in connected_states or t not in connected_tasks:
            continue
        graph.add_edge(t, s, rho=v["rho"])
    return stn, graph


def draw_graph(stn, graph, with_labels=False, verbose=False):
    # Determine initial and final states
    initial_states = {
        state
        for state, attr in stn["STATES"].items()
        if attr["initial"] > 0 and graph.in_degree(state) == 0
    }
    final_states = {state for state in stn["STATES"] if graph.out_degree(state) == 0}
    if verbose:
        print(f"initial states: {initial_states}")
        print(f"final states: {final_states}")

    tasks = set()
    for s, t in stn["ST_ARCS"]:
        tasks.add(t)
    for t, s in stn["TS_ARCS"]:
        tasks.add(t)

    # Position nodes
    initial_x = 0.1
    intermediate_x = 0.5
    final_x = 0.9
    y = 0.5
    new_pos = {}
    for node in graph.nodes():
        if node in initial_states:
            new_pos[node] = (initial_x, y)
            initial_x += 0.1
        elif node in final_states:
            new_pos[node] = (final_x, y)
            final_x -= 0.1
        else:
            new_pos[node] = (intermediate_x, y)
            intermediate_x += 0.1

    # Adjust positions of tasks based on their connected states
    task_y_offset = 0.2
    for node in tasks:
        avg_x = sum(
            new_pos[pred][0] for pred in graph.predecessors(node)
        ) / graph.in_degree(node)
        new_pos[node] = (avg_x, y - task_y_offset)

    # Draw the network graph
    plt.figure(figsize=(14, 8))
    node_colors = [
        (
            "yellow"
            if n in initial_states
            else (
                "red"
                if n in final_states
                else "skyblue" if graph.nodes[n]["type"] == "state" else "lightgreen"
            )
        )
        for n in graph.nodes()
    ]
    nx.draw(
        graph,
        new_pos,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=100,
        font_size=9,
        font_weight="bold",
        arrowstyle="-|>",
        arrowsize=20,
    )
    edge_labels = nx.get_edge_attributes(graph, "rho")
    if with_labels:
        nx.draw_networkx_edge_labels(
            graph, new_pos, edge_labels=edge_labels, font_color="red"
        )
    plt.title("Network Graph with Spatially Separated Initial and Final States")
    st.pyplot(plt)  # plt.show()


def list_predecessors(stn, graph, node, visited=set(), depth=0, verbose=False):
    visited.add(node)
    pred = list(graph.predecessors(node))
    if verbose:
        print(
            "  " * depth,
            node,
            graph.nodes[node]["type"],
            "->",
            [(p, graph.edges[p, node]["rho"]) for p in pred],
        )
    for p in pred:
        list_predecessors(stn, graph, p, visited, depth + 1, verbose)
    return visited


def clean_stn(stn, graph, target, verbose=False):
    visited = list_predecessors(stn, graph, target, verbose=verbose)
    return {
        # time grid
        "TIME": stn["TIME"],
        # states
        "STATES": {s: stn["STATES"][s] for s in stn["STATES"] if s in visited},
        # state-to-task arcs indexed by (state, task)
        "ST_ARCS": {
            (s, t): stn["ST_ARCS"][(s, t)]
            for (s, t) in stn["ST_ARCS"]
            if s in visited and t in visited
        },
        # task-to-state arcs indexed by (task, state)
        "TS_ARCS": {
            (t, s): stn["TS_ARCS"][(t, s)]
            for (t, s) in stn["TS_ARCS"]
            if s in visited and t in visited
        },
        # unit data indexed by (unit, task)
        "UNIT_TASKS": {
            (u, t): stn["UNIT_TASKS"][(u, t)]
            for (u, t) in stn["UNIT_TASKS"]
            if t in visited
        },
    }
