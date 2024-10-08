import streamlit as st
import pandas as pd
import random
import itertools
import os
from amplpy import AMPL
from .solutions import solutions
from ..common import solver_selector, MP_SOLVERS_LINKS

BASE_MODEL = r"""
# The base number of this sudoku; 3 is the default (9 numbers game)
param BaseNumber default 3;

# The length of each line/column, derived from BaseNumber
param GridSize := BaseNumber * BaseNumber;

# Set of all Rows
set Rows := 1..GridSize;

# Set of all Columns
set Columns := 1..GridSize;

# This indexed set memorizes the tuples of coordinates for each 
# sub-grid making up the sudoku grid
set SubGridCoordinates{SubGridRow in 1..BaseNumber, SubGridCol in 1..BaseNumber} 
    within {Rows, Columns}
    = {(SubGridRow-1)*BaseNumber+1..SubGridRow*BaseNumber, 
        (SubGridCol-1)*BaseNumber+1..SubGridCol*BaseNumber};

# The variables representing the numbers at all positions in the sudoku grid
var SudokuGrid{Rows, Columns} >= 1, <= GridSize integer;

# Set this parameter to non-zero to fix a position to have that value
param FixedValues{Rows, Columns} default 0;

# Dummy objective, just to "encourage" the solver to have a defined objective
maximize DummyObjective: SudokuGrid[1,1];

# Fix input data (forces the variable at the corresponding location to have
# the same value as the parameter)
subject to FixFixedValues{row in Rows, col in Columns : FixedValues[row, col] > 0}: 
    SudokuGrid[row, col] = FixedValues[row, col];
"""

MIP_MODEL = r"""
# Binary decision variable that represents the presence or absence of a specific 
# number at a particular position in the Sudoku grid.
var NumberPresence{Number in 1..GridSize, Columns, Rows} binary;

# Each position in the grid must have only one number
subject to MIPUniqueNumberPerPosition{row in Rows, col in Columns}: 
    sum{num in 1..GridSize} NumberPresence[num, col, row] = 1;

# Each number must appear exactly once in each row
subject to MIPUniqueNumberPerRow{row in Rows, num in 1..GridSize}: 
    sum{col in Columns} NumberPresence[num, col, row] = 1;

# Each number must appear exactly once in each column
subject to MIPUniqueNumberPerColumn{col in Columns, num in 1..GridSize}: 
    sum{row in Rows} NumberPresence[num, col, row] = 1;

# Each number must appear exactly once in each sub-grid
subject to MIPUniqueNumberPerSubGrid{num in 1..GridSize, SubGridRow in 1..BaseNumber, SubGridCol in 1..BaseNumber}: 
    sum{(row, col) in SubGridCoordinates[SubGridRow, SubGridCol]}
    NumberPresence[num, col, row] = 1;

# Link to the SudokuGrid variable
subject to MIPLinkToSudokuGrid{row in Rows, col in Columns}: 
    sum{num in 1..GridSize} NumberPresence[num, col, row] * num = SudokuGrid[row, col];
"""

CP_MODEL = r"""
# All numbers in one row must be different
subject to RowAllDifferent{row in Rows}:   
    alldiff {col in Columns} SudokuGrid[row, col];

# All numbers in one column must be different
subject to ColumnAllDifferent{col in Columns}:   
    alldiff {row in Rows} SudokuGrid[row, col];

# All numbers within each sub-grid must be different
subject to SubGridAllDifferent{SubGridRow in 1..BaseNumber, SubGridCol in 1..BaseNumber}: 
    alldiff {(row, col) in SubGridCoordinates[SubGridRow, SubGridCol]} SudokuGrid[row, col];
"""


@st.cache_data
def solve_sudoku(base=3, grid=None, model="cp", solver="gurobi"):
    ampl = AMPL()
    ampl.eval(BASE_MODEL)
    ampl.param["BaseNumber"] = base

    if grid is not None:
        ps = grid.stack()
        ps.index = ps.index.map(lambda x: tuple(map(int, x)))
        ps = ps.apply(lambda v: 0 if v == " " else int(v))
        ampl.param["FixedValues"] = ps

    if model == "cp":
        ampl.eval(CP_MODEL)
    else:
        ampl.eval(MIP_MODEL)

    output = ampl.solve(solver=solver, mp_options="outlev=1", return_output=True)
    solve_time = ampl.get_value("_solve_time")

    solution = ampl.var["SudokuGrid"].to_pandas().unstack()
    solution.columns = solution.columns.droplevel()
    solution.rename_axis("Grid", inplace=True)
    return output, solve_time, solution


def permute_sudoku(board):
    base = int(len(board) ** 0.5)
    row_permutations = list(itertools.permutations(range(base)))
    col_permutations = list(itertools.permutations(range(base)))
    generated_boards = []
    for row_perm in row_permutations:
        for col_perm in col_permutations:
            new_board = []
            for i in range(base):
                for j in row_perm:
                    new_row = []
                    for k in range(base):
                        for l in col_perm:
                            new_row.append(board[i * base + j][k * base + l])
                    new_board.append(new_row)
            generated_boards.append(new_board)

    return generated_boards


def solution_to_df(solution):
    grid_size = len(solution)
    return pd.DataFrame(
        solution, index=range(1, 1 + grid_size), columns=range(1, 1 + grid_size)
    )


def save_solutions(max_base=4):
    solutions = {}
    for i in range(3, max_base + 1):
        _, _, solution = solve_sudoku(i)
        solutions[i] = solution.values.tolist()

    with open(os.path.join(os.path.dirname(__file__), "solutions.py"), "w") as f:
        print(f"solutions = {solutions}", file=f)


@st.cache_data
def generate_random_grid(solution, n_missing):
    base = int(len(solution) ** 0.5)
    grid_size = base * base
    solution = solution.values.tolist()
    solution = random.choice(permute_sudoku(solution))
    missing = set(
        random.sample(
            [(i, j) for i in range(grid_size) for j in range(grid_size)],
            min(n_missing, base**4),
        )
    )
    data = [
        [solution[i][j] if (i, j) not in missing else " " for i in range(grid_size)]
        for j in range(grid_size)
    ]
    return pd.DataFrame(
        data, index=range(1, 1 + grid_size), columns=range(1, 1 + grid_size)
    )


def main():
    st.title("🔢 Sudoku Solver")
    st.markdown(
        """
        Simple sudoku model with two formulations: as a **Constraint Programming (CP)** problem using the `alldiff`
        operator and as **Mixed Integer Programming (MIP)**.
        """
    )
    st.info(
        f"""
        The `alldiff` operator needs a solver supporting logical constraints or a MIP solver with  
        [AMPL's Automatic Reformulation Support](https://mp.ampl.com/model-guide.html) such as: {MP_SOLVERS_LINKS}.
        """
    )

    sudoku_base = st.slider(
        "Base of the sudoku puzzle 👇",
        help="For a normal sudoku game, where the numbers go from 1 to 9, this has to be set to 3.",
        min_value=3,
        max_value=5,
        step=1,
        value=3,
    )
    grid_size = sudoku_base**2

    # save_solutions(max_base=5)

    if sudoku_base in solutions:
        sudoku_solution_df = solution_to_df(solutions[sudoku_base])
    else:
        _, _, sudoku_solution_df = solve_sudoku(sudoku_base)

    max_missing = sudoku_base**4
    sudoku_missing = st.slider(
        "Number of missing values 👇",
        min_value=0,
        max_value=min(100, max_missing),
        step=1,
        value=25,
    )

    st.session_state.sudoku_grid = generate_random_grid(
        sudoku_solution_df, sudoku_missing
    )

    st.session_state.sudoku_grid = st.data_editor(
        st.session_state.sudoku_grid,
        key="sudoku_grid_widget",
        column_config={
            f"{i}": st.column_config.SelectboxColumn(
                options=list(range(1, 1 + grid_size)) + [" "],
                default="",
            )
            for i in range(1, 1 + grid_size)
        },
    )

    st.markdown("## Constraint Programming Model")

    st.code(BASE_MODEL + CP_MODEL)

    st.markdown("## Mixed Integer Programming Model")

    st.code(BASE_MODEL + MIP_MODEL)

    st.markdown("## Solve")

    model_types = ["Constraint Programming model", "Mixed Integer Programming model"]
    model_selected = st.selectbox("Pick the model to use 👇", model_types, key="model")
    if model_types.index(model_selected) == 0:
        model_type = "cp"
    else:
        model_type = "mip"

    # Select the solver to use
    solver, _ = solver_selector(mp_only=True)

    # Solve
    output, solve_time, solution = solve_sudoku(
        sudoku_base, st.session_state.sudoku_grid, model_type, solver
    )

    st.markdown("### Solution")
    st.markdown(
        f"**Solved in {solve_time:.2f} seconds with {solver} using {model_selected}.**"
    )
    st.write(solution)

    st.markdown("### MIP Solver output")
    st.markdown(f"```\n{output}\n```")

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/sudoku)] [[Google Colab Notebook](https://ampl.com/colab/notebooks/simple-sudoku-solver-using-logical-constraints-with-gui.html)]"""
    )
