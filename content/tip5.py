import streamlit as st

title = "Tip #5: Non-contiguous variable domains"


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

    ```
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

    # 2. Finite domains as unions of integer sets:
    # var Buy {FOOD} in 0..4 union 9..13 union 17..20;

    # 3. A union of a number and an interval (semi-continuous variables):
    # var Buy {FOOD} in {0} union interval [10, 50];

    # 4. A union of two intervals
    # var Buy {FOOD} in interval[2, 5] union interval[7,10];

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

    option solver highs;
    solve; display Buy;
    ```

    Running the example above (solving with HiGHS, Gurobi, and Knitro) we get the following output:
    1. With `var Buy {FOOD} in {0, 10, 30, 45, 55};` and HIGHS:
    ```
    HiGHS 1.4.0: optimal solution; objective 96.5
    0 simplex iterations
    1 branching nodes
    absmipgap=1.42109e-14, relmipgap=0
    Buy [*] :=
    BEEF   0
    CHK   0
    FISH   0
    HAM   0
    MCH  30
    MTL  10
    SPG  10
    TUR   0
    ;
    ```
    2. With `var Buy {FOOD} in 0..4 union 9..13 union 17..20;`, and Gurobi:

    ```
    Gurobi 10.0.0: optimal solution; objective 89.79
    51 simplex iterations
    1 branching nodes
    Buy [*] :=
    BEEF   0
    CHK  17
    FISH   0
    HAM   0
    MCH  20
    MTL   4
    SPG   0
    TUR   0
    ;
    ```
    3. With `var Buy {FOOD} in {0} union interval [10, 50];`, and Knitro:
    ```
    Knitro 13.2.0: Locally optimal or satisfactory solution.
    objective 88.2; optimality gap -1.66e-11
    1 nodes; 0 subproblem solves
    Buy [*] :=
    BEEF   0
    CHK   0
    FISH   0
    HAM   0
    MCH  46.6667
    MTL   0
    SPG   0
    TUR   0
    ;
    ```
    4. With `var Buy {FOOD} in interval[2, 5] union interval[7,10];` and XPRESS:
    ```
    xpress 41.01.01: optimal solution; objective 101.0133333
    6 simplex iterations
    1 branching nodes
    Buy [*] :=
    BEEF   2
    CHK  10
    FISH   2
    HAM   2
    MCH  10
    MTL  10
    SPG   7.33333
    TUR   2
    ;
    ```

    Documentation of the extended AMPL modeling syntax is available in the [MP Modeling Guide](https://amplmp.readthedocs.io/en/latest/rst/model-guide.html) - [Set membership operator](https://amplmp.readthedocs.io/en/latest/rst/modeling-expressions.html#set-membership-operator).
        """
    )
