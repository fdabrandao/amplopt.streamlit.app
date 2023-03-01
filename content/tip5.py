import streamlit as st
from .utils import snippet

title = "Tip #5: Non-contiguous variable domains"

diet_dat = """
    data;
    set NUTR := A B1 B2 C ;
    set FOOD := BEEF CHK FISH HAM MCH MTL SPG TUR ;
    param:   cost  f_min  f_max :=
    BEEF   3.19    0     100
    CHK    2.59    0     100
    FISH   2.29    0     100
    HAM    2.89    0     100
    MCH    1.89    0     100
    MTL    1.99    0     100
    SPG    1.99    0     100
    TUR    2.49    0     100 ;
    param:   n_min  n_max :=
    A      700   10000
    C      700   10000
    B1     700   10000
    B2     700   10000 ;
    param amt (tr):
            A    C   B1   B2 :=
    BEEF   60   20   10   15
    CHK     8    0   20   20
    FISH    8   10   15   10
    HAM    40   40   35   10
    MCH    15   35   15   15
    MTL    70   30   15   15
    SPG    25   50   25   15
    TUR    60   20   15   10 ;
    model;
"""


def run():
    st.markdown(
        """
    Sometimes variable values are restricted to non-contiguous sets.

    - Finite domain:
    ```
    var Buy {FOOD} in {0, 10, 30, 45, 55};
    ```
    - Finite domains as unions of integer sets:
    ```
    var Buy {FOOD} in 0..4 union 9..13 union 17..20;
    ```
    - A union of a number and an interval (semi-continuous variables):
    ```
    var Buy {FOOD} in {0} union interval [10, 50];
    ```
    - A union of two intervals:
    ```
    var Buy {FOOD} in interval[2, 5] union interval[7,10];
    ```

    This works with any solver supporting integer variables.
    Try on Google Colab by modifying the diet example:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ampl/amplcolab/blob/master/ampl-book/diet.ipynb)

    Full example using a diet model modified with `var Buy {FOOD} in interval[2, 5] union interval[7,10];` instead of the original declaration `var Buy {j in FOOD} >= f_min[j], <= f_max[j];`:
    """
    )

    snippet(
        """fullexample""",
        """
        set NUTR;
        set FOOD;
        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];
        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];
        param amt {NUTR,FOOD} >= 0;

        # 4. A union of two intervals
        var Buy {FOOD} in interval[2, 5] union interval[7,10];

        minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];
        subject to Diet {i in NUTR}:
        n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];

        data;
        set NUTR := A B1 B2 C ;
        set FOOD := BEEF CHK FISH HAM MCH MTL SPG TUR ;
        param:   cost  f_min  f_max :=
        BEEF   3.19    0     100
        CHK    2.59    0     100
        FISH   2.29    0     100
        HAM    2.89    0     100
        MCH    1.89    0     100
        MTL    1.99    0     100
        SPG    1.99    0     100
        TUR    2.49    0     100 ;
        param:   n_min  n_max :=
        A      700   10000
        C      700   10000
        B1     700   10000
        B2     700   10000 ;
        param amt (tr):
                A    C   B1   B2 :=
        BEEF   60   20   10   15
        CHK     8    0   20   20
        FISH    8   10   15   10
        HAM    40   40   35   10
        MCH    15   35   15   15
        MTL    70   30   15   15
        SPG    25   50   25   15
        TUR    60   20   15   10 ;
        model;
        """,
        """
        option solver highs; solve;
        display Buy;
        """,
    )

    st.markdown(
        """
    Running the example above (solving with HiGHS, Gurobi, CBC, and XPRESS) we get the following output:
    1. With `var Buy {FOOD} in {0, 10, 30, 45, 55};` and HIGHS:"""
    )
    snippet(
        """example1""",
        """
        set NUTR;
        set FOOD;
        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];
        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];
        param amt {NUTR,FOOD} >= 0;

        # 1. Finite domain:
        var Buy {FOOD} in {0, 10, 30, 45, 55};

        minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];
        subject to Diet {i in NUTR}:
        n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
        """,
        """
        option solver highs; solve;
        display Buy;
        """,
        data=diet_dat,
    )

    st.markdown(
        "2. With `var Buy {FOOD} in 0..4 union 9..13 union 17..20;`, and Gurobi:"
    )
    snippet(
        """example2""",
        """
        set NUTR;
        set FOOD;
        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];
        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];
        param amt {NUTR,FOOD} >= 0;

        # 2. Finite domains as unions of integer sets:
        var Buy {FOOD} in 0..4 union 9..13 union 17..20;

        minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];
        subject to Diet {i in NUTR}:
        n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
        """,
        """
        option solver gurobi; solve;
        display Buy;
        """,
        data=diet_dat,
    )

    st.markdown("3. With `var Buy {FOOD} in {0} union interval [10, 50];`, and CBC:")
    snippet(
        """example3""",
        """
        set NUTR;
        set FOOD;
        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];
        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];
        param amt {NUTR,FOOD} >= 0;

        # 3. A union of a number and an interval (semi-continuous variables):
        var Buy {FOOD} in {0} union interval [10, 50];

        minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];
        subject to Diet {i in NUTR}:
        n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
        """,
        """
        option solver cbc; solve;
        display Buy;
        """,
        data=diet_dat,
    )

    st.markdown(
        "4. With `var Buy {FOOD} in interval[2, 5] union interval[7,10];` and XPRESS:"
    )
    snippet(
        """example4""",
        """
        set NUTR;
        set FOOD;
        param cost {FOOD} > 0;
        param f_min {FOOD} >= 0;
        param f_max {j in FOOD} >= f_min[j];
        param n_min {NUTR} >= 0;
        param n_max {i in NUTR} >= n_min[i];
        param amt {NUTR,FOOD} >= 0;

        # 4. A union of two intervals
        var Buy {FOOD} in interval[2, 5] union interval[7,10];

        minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];
        subject to Diet {i in NUTR}:
        n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
        """,
        """
        option solver xpress; solve;
        display Buy;
        """,
        data=diet_dat,
    )
    st.markdown(
        """
    Documentation of the extended AMPL modeling syntax is available in the [MP Modeling Guide](https://amplmp.readthedocs.io/en/latest/rst/model-guide.html) - [Set membership operator](https://amplmp.readthedocs.io/en/latest/rst/modeling-expressions.html#set-membership-operator).
        """
    )
