import streamlit as st
from .utils import snippet, MPSOLVERS

title = "Tip #2: Equivalence"


def run():
    st.markdown(
        """
    **“Of two variables x and y, exactly one should be positive”**: how to model this constraint? This is a stricter condition that in the previous [Tip #1](?tip=1).

    For the new [MP Library-based drivers](https://amplmp.readthedocs.io/en/latest/rst/drivers.html) (e.g., [gurobi](https://ampl.com/products/solvers/solvers-we-sell/gurobi/), [highs](https://ampl.com/products/solvers/open-source-solvers/), [cbc](https://ampl.com/products/solvers/open-source-solvers/), [copt](https://ampl.com/products/solvers/solvers-we-sell/copt/), [xpress](https://ampl.com/products/solvers/solvers-we-sell/xpress/)), as well as for Constraint Programming solvers (ilogcp, gecode, jacop), this goes via AMPL logical operators:

    `x > 0 <==> y <= 0`

    We can ask what `x > 0` means for non-integer variables. It will be replaced by a non-strict inequality with a small tolerance: `x >= eps`.

    If a specific “gap” is desired, for example if we want to exclude numbers in the interval (0, 3), use a disjunctive normal form (DNF):

    `(x <= 0 and y >= 3) or (x >= 3 and y <= 0)`

    #### Small complete examples:
    """
    )

    st.markdown("1. With MP, using the `<==>` operator:")
    snippet(
        """example1""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        minimize total: 5 * x + 2 * y;
        s.t. exactly_one_positive: x > 0 <==> y <= 0;
        """,
        """
        option solver $SOLVER; solve;
        display x, y;
        """,
        solvers=MPSOLVERS,
    )

    st.markdown("2. With MP, using DNF to exclude a gap interval:")
    snippet(
        """example2""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        minimize total: 5 * x + 2 * y;
        s.t. exactly_one_positive_with_gap:
            (x <= 0 and y >= 3) or (x >= 3 and y <= 0);
        """,
        """
        option solver $SOLVER; solve;
        display x, y;
        """,
        solvers=MPSOLVERS,
    )

    st.markdown("3. Without MP you would need to linearize the logic using big-M:")
    snippet(
        """example3""",
        """
        var x >= -1000 <= 1000;
        var y >= -1000 <= 1000;
        var b binary;
        minimize total: 5 * x + 2 * y;
        s.t. big_m_1_x: x <= b * 1000;
        s.t. big_m_1_y: y >= 3 - b * 1003;
        s.t. big_m_2_x: x >= -1000 + b * 1003;
        s.t. big_m_2_y: y <= (1-b) * 1000;
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
