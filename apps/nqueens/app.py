import streamlit as st
from amplpy import AMPL


def main():
    st.title("👑 N-Queens")
    st.markdown(
        """
        **How can $n$ queens be placed on an $n \\times n$ chessboard so that no two of them attack each other?**

        Constraint `alldiff` enforces a set of integer variables to take distinct values. Using `alldiff`, we can model N-Queens as follows:

        ```ampl
        param n integer > 0; # N-queens
        var Row {1..n} integer >= 1 <= n;
        s.t. row_attacks: alldiff ({j in 1..n} Row[j]);
        s.t. diag_attacks: alldiff ({j in 1..n} Row[j]+j);
        s.t. rdiag_attacks: alldiff ({j in 1..n} Row[j]-j);
        ```
        """
    )

    ampl = AMPL()
    ampl.eval(
        """
    param n integer > 0; # N-queens
    var Row {1..n} integer >= 1 <= n;
    s.t. row_attacks: alldiff ({j in 1..n} Row[j]);
    s.t. diag_attacks: alldiff ({j in 1..n} Row[j]+j);
    s.t. rdiag_attacks: alldiff ({j in 1..n} Row[j]-j);
    """
    )
    ampl.option["solver"] = "highs"
    ampl.option["highs_options"] = "outlev=1"

    n = st.slider("How many queens?", 2, 25, 8)

    ampl.param["n"] = n
    output = ampl.get_output("solve;")
    solution = ampl.get_data("Row").to_dict()
    queens = set((int(r) - 1, int(c) - 1) for c, r in solution.items())

    st.write("### Solution")
    solution = "#" + " # " * (n) + "#\n"
    for r in range(n):
        row = "".join([" Q " if (r, c) in queens else " + " for c in range(n)])
        # st.write(f"`{row}`")
        solution += "#" + row + "#\n"
    solution += "#" + " # " * (n) + "#\n"
    st.write(f"```\n{solution}\n```")

    st.write("# HiGHS output")
    st.write(f"```\n{output}\n```")

    st.markdown(
        """

    ## AMPL :heart: Python :heart: Streamlit

    **Deploy optimization apps to [Streamlit Cloud](https://streamlit.io/) with AMPL**

    [AMPL and all Solvers are now available as Python Packages.](https://dev.ampl.com/ampl/python/)
    To use them in Streamlit you just need to list the modules in the [requirements.txt](https://github.com/fdabrandao/streamlit-nqueens/blob/master/requirements.txt) file as follows:
    ```
    --index-url https://pypi.ampl.com # AMPL's Python Package Index
    --extra-index-url https://pypi.org/simple
    ampl_module_base # AMPL
    ampl_module_highs # HiGHS solver
    ampl_module_gurobi # Gurobi solver
    amplpy # Python API for AMPL
    ```

    and then just load them in [streamlit_app.py](https://github.com/fdabrandao/streamlit-nqueens/blob/master/streamlit_app.py):
    ```python
    from amplpy import AMPL
    ampl = AMPL()
    ```

    - GitHub repository for this app: https://github.com/fdabrandao/streamlit-nqueens
    - Python API documentation: https://amplpy.readthedocs.io
    - Python modules documentation: https://dev.ampl.com/ampl/python/
    """
    )

    st.markdown(
        """
    #### [[On LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7024414805193089024)] [[On Twitter](https://twitter.com/AMPLopt/status/1618648831230451715)]  [[Source Code on GitHub](https://github.com/fdabrandao/streamlit-nqueens)]
    """
    )
