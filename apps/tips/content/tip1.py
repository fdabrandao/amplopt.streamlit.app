import streamlit as st
from .utils import snippet, MPSOLVERS

title = "Tip #1: Disjunctions"


def run():
    st.markdown(
        """
    **“Two variables x and y cannot be positive at the same time”**: how to model this constraint? For the new [MP Library-based drivers](https://amplmp.readthedocs.io/en/latest/rst/drivers.html) (e.g., [gurobi](https://ampl.com/products/solvers/solvers-we-sell/gurobi/), [highs](https://ampl.com/products/solvers/open-source-solvers/), [xpress](https://ampl.com/products/solvers/solvers-we-sell/xpress/), [cbc](https://ampl.com/products/solvers/open-source-solvers/), [copt](https://ampl.com/products/solvers/solvers-we-sell/copt/)), as well as for Constraint Programming solvers (ilogcp, gecode, jacop), this goes via AMPL logical operators:

    `x <= 0 or y <= 0`

    #### Small complete examples:
    """
    )

    st.markdown("1. With MP, using the `or` operator:")
    snippet(
        """example1""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        maximize total: 5 * x + 2 * y;
        s.t. only_one_positive: x <= 0 or y <= 0;
        """,
        """
        option solver $SOLVER; solve;
        display x, y;
        """,
        solvers=MPSOLVERS,
    )

    st.markdown("2. With MP, using implication:")
    snippet(
        """example2""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        maximize total: 5 * x + 2 * y;
        s.t. only_one_positive: x > 0 ==> y <= 0;
        """,
        """
        option solver $SOLVER; solve;
        display x, y;
        """,
        solvers=MPSOLVERS,
    )

    st.markdown("3. Without MP you would need to linearize the constraint using big-M:")
    snippet(
        """example3""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        var b binary;
        maximize total: 5 * x + 2 * y;
        s.t. big_m_1: x <= b * 1000;
        s.t. big_m_2: y <= (1-b) * 1000;
        """,
        """
        option solver $SOLVER; solve;
        display x, y;
        """,
        solvers=MPSOLVERS,
    )

    st.markdown(
        """
    More examples and documentation are in the [MP Modeling Guide](https://amplmp.readthedocs.io/en/latest/rst/model-guide.html).
    """
    )
