import streamlit as st
import pandas as pd
import random
import itertools
from amplpy import AMPL


def permute_sudoku(board):
    row_permutations = list(itertools.permutations([0, 1, 2]))
    col_permutations = list(itertools.permutations([0, 1, 2]))
    generated_boards = []
    for row_perm in row_permutations:
        for col_perm in col_permutations:
            new_board = []
            for i in range(3):
                for j in row_perm:
                    new_row = []
                    for k in range(3):
                        for l in col_perm:
                            new_row.append(board[i * 3 + j][k * 3 + l])
                    new_board.append(new_row)
            generated_boards.append(new_board)

    return generated_boards


def random_state(prob):
    solution = [
        [2, 5, 7, 8, 6, 3, 1, 4, 9],
        [4, 9, 6, 5, 7, 1, 8, 3, 2],
        [8, 1, 3, 9, 4, 2, 7, 6, 5],
        [1, 6, 5, 2, 9, 4, 3, 7, 8],
        [9, 8, 4, 1, 3, 7, 5, 2, 6],
        [3, 7, 2, 6, 5, 8, 4, 9, 1],
        [7, 2, 9, 4, 8, 5, 6, 1, 3],
        [5, 3, 1, 7, 2, 6, 9, 8, 4],
        [6, 4, 8, 3, 1, 9, 2, 5, 7],
    ]
    solution = random.choice(permute_sudoku(solution))
    data = [
        [solution[i][j] if random.random() < prob else " " for i in range(9)]
        for j in range(9)
    ]
    return pd.DataFrame(data, index=range(1, 10), columns=range(1, 10))


def main():
    st.markdown(
        """
    # 🔢 Sudoku Solver

    """
    )

    prob = st.slider(
        "Generate a random board with the following probability of having a value in a cell 👇",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=1 / 3.0,
    )
    if prob != st.session_state.get("sudoku_prob", None):
        st.session_state.sudoku_prob = prob
        st.session_state.sudoku_grid = random_state(prob)

    st.session_state.sudoku_grid = st.data_editor(
        st.session_state.sudoku_grid,
        key="sudoku_grid_widget",
        column_config={
            f"{i}": st.column_config.SelectboxColumn(
                options=list(range(1, 10)) + [" "],
                default="",
            )
            for i in range(1, 10)
        },
    )

    base_model = r"""
    # The base number of this sudoku; 3 is the default (9 numbers game)
    param BaseNumber default 3;

    # The length of each line/column, derived from BaseNumber
    param GridSize := BaseNumber * BaseNumber;

    # Set of all Rows
    set Rows := {1..GridSize};

    # Set of all Columns
    set Columns := {1..GridSize};

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

    mip_model = r"""
    # Binary decision variable that represents the presence or absence of a specific 
    # number at a particular position in the Sudoku grid.
    var NumberPresence{Number in 1..GridSize, Columns, Rows} binary;

    # Each position in the grid must have only one number
    MIPUniqueNumberPerPosition{row in Rows, col in Columns}: 
        sum{num in 1..GridSize} NumberPresence[num, col, row] = 1;

    # Each number must appear exactly once in each row
    MIPUniqueNumberPerRow{row in Rows, num in 1..GridSize}: 
        sum{col in Columns} NumberPresence[num, col, row] = 1;

    # Each number must appear exactly once in each column
    MIPUniqueNumberPerColumn{col in Columns, num in 1..GridSize}: 
        sum{row in Rows} NumberPresence[num, col, row] = 1;

    # Each number must appear exactly once in each sub-grid
    MIPUniqueNumberPerSubGrid{num in 1..GridSize, SubGridRow in 1..BaseNumber, SubGridCol in 1..BaseNumber}: 
        sum{(row, col) in SubGridCoordinates[SubGridRow, SubGridCol]}
        NumberPresence[num, col, row] = 1;

    # Link to the SudokuGrid variable
    MIPLinkToSudokuGrid{row in Rows, col in Columns}: 
        sum{num in 1..GridSize} NumberPresence[num, col, row] * num = SudokuGrid[row, col];
    """

    cp_model = r"""
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

    st.markdown("## Constraint Programming Model")

    st.code(base_model + cp_model)

    st.markdown("## Mixed Integer Programming Model")

    st.code(base_model + mip_model)

    st.markdown("## Solve")

    ampl = AMPL()
    ampl.eval(base_model)
    ampl.param["BaseNumber"] = 3

    ps = st.session_state.sudoku_grid.stack()
    ps.index = ps.index.map(lambda x: tuple(map(int, x)))
    ps = ps.apply(lambda v: 0 if v == " " else int(v))
    ampl.param["FixedValues"] = ps

    model_types = ["Constraint Programming model", "Mixed Integer Programming model"]
    model_selected = st.selectbox("Pick the model to use 👇", model_types, key="model")
    if model_types.index(model_selected) == 0:
        ampl.eval(cp_model)
    else:
        ampl.eval(mip_model)

    solvers = ["gurobi", "xpress", "cplex", "mosek", "copt", "highs", "scip", "cbc"]
    solver = st.selectbox("Pick the MIP solver to use 👇", solvers, key="solver")
    if solver == "cplex":
        solver = "cplexmp"

    output = ampl.solve(solver=solver, mp_options="outlev=1", return_output=True)

    st.markdown("### Solution")

    solution = ampl.var["SudokuGrid"].to_pandas()
    df = solution.unstack()
    df.columns = df.columns.droplevel()
    df.rename_axis("Grid", inplace=True)
    st.write(df)

    st.markdown("### MIP Solver output")

    st.write(f"```\n{output}\n```")
