import streamlit as st
from .utils import snippet

title = "Tip #6: Robust Linear Programming with Ellipsoidal Uncertainty"


def run():
    st.markdown(
        """
**Sometimes values are not known exactly!**
"""
    )

    st.image("static/apps/tips/tip6_feasible_region.png")

    st.markdown(
        """
In the diet problem we want to find a diet that satisfies certain nutricial requiments while also minimizing the total cost. **What if the costs were not know exactly?**

One simple approach is via **robust optimization** with **ellipsoidal uncertainty** as follows:  
```python
var t >= 0; # Auxiliary variable
minimize Total_Cost:
sum {j in FOOD} cost[j] * Buy[j] + t; # Added to the objective
subject to Ellipsoid:                     
t >= sqrt(sum {j in FOOD} (0.4 * cost[j] * Buy[j])^2);
                # Second-order cone
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ampl/amplcolab/blob/master/authors/glebbelov/modeling-tips/tip6_robust_linear_programming.ipynb)

# Simplified diet problem

Let's consider a simplified version of the diet problem and let's consider uncertainty:
- We have just two types of food
- We just want to satisfy the required number of calories per day
- **The costs are not known exactly**

If the costs were known exactly, we could model this problem as follows:
```python
set NUTR;
set FOOD;

param cost {FOOD} > 0;
param calories {FOOD} >= 0;
param min_calories;
param max_calories;

var Buy {j in FOOD} >= 0;

minimize Total_Cost:
sum {j in FOOD} cost[j] * Buy[j];

subject to Required_Calories:
    min_calories <= sum {i in FOOD} calories[i] * Buy[i] <= max_calories;
```

Since there is uncertainty we can do the following modifications:

```python
var t >= 0; # Auxiliary variable
minimize Total_Cost:
sum {j in FOOD} cost[j] * Buy[j] + t; # Added to the objective
subject to Ellipsoid:                     
t >= sqrt(sum {j in FOOD} (0.4 * cost[j] * Buy[j])^2); # Second-order cone
````
"""
    )

    st.markdown("Complete model:")
    snippet(
        """fullexample""",
        """
        set NUTR;
        set FOOD;

        param cost {FOOD} > 0;
        param calories {FOOD} >= 0;
        param min_calories;
        param max_calories;
        param robust default 1;

        var Buy {j in FOOD} >= 0;
        var t >= 0; # Auxiliary variable

        minimize Total_Cost:
            sum {j in FOOD} cost[j] * Buy[j] + t; # Added to the objective

        subject to Required_Calories:
            min_calories <= sum {i in FOOD} calories[i] * Buy[i] <= max_calories;

        subject to Ellipsoid{if robust}:                
            t >= sqrt(sum {j in FOOD} (0.4 * cost[j] * Buy[j])^2); # Second-order cone
        """,
        """
        printf "> Not robust:\\n";
        option solver $SOLVER;
        let robust := 0;
        solve;
        display Buy, Total_Cost;
        printf "> Robust:\\n";
        let robust := 1;
        solve;
        display Buy, Total_Cost - t;
        """,
        data_code="""
        ampl.set["FOOD"] = ["BEEF", "CHK"]
        ampl.param["cost"] = {"BEEF": 1, "CHK": 1}
        ampl.param["min_calories"] = 2000
        ampl.param["max_calories"] = 2500
        ampl.param["calories"] = {"BEEF": 250, "CHK": 239}
        """,
        solvers=["mosek", "gurobi"],
    )

    st.markdown(
        """
- Ellipsoidal uncertainty is of the less conservative kind: [Introduction](https://docs.mosek.com/latest/toolbox/case-studies-robust-lo.html).

- Documentation on AMPL conic and extended modeling can be found in the [MP Modeling Guide](https://amplmp.readthedocs.io/en/latest/rst/model-guide.html).
        """
    )
