import streamlit as st
from .utils import snippet

title = "Tip #8: Absolute assignment `y = abs(x)`"


def run():
    st.markdown(
        """
        ### In AMPL in can simply write `y = abs(x)`. How is it linearized?

        *   This is equivalent to `y >= abs(x)` and `y <= abs(x)` which have **very different linearizations!**
            
        *   **AMPL MP reformulates this automatically for you** according to what each solver supports!
            
            *   For `y >= abs(x)` we just need the two constraints `(y >= x) and (y >= -x)`
                
            *   For `y <= abs(x)` we need the disjunctive constraint  
                `(y <= x) or (y <= -x)`  
                which can be further reformulated as follows:  
                `B1 ==> (y <= x)`  
                `B2 ==> (y <= -x)`  
                `B1 + B2 >= 1`  
                where indicators can be linearized using Big-M.
                
        *   `x` **and** `y` **can be just variables or more general expressions!**
            

        ## Example

        ```ampl
        var x {1..2} >=-30 <=100;
        minimize Objective: abs(x[1]) - 2*abs(x[2]);
        s.t. Con1: 3*x[1] - 2*x[2] <=  8;
        s.t. Con2: x[1] +   x[2] == 14;
        ```

        Adding variables `U[1]` and `U[2]`, we can change the objective to `U[1] - 2*U[2]`,
        with `U[1] >= abs(x[1])` and `U[2] <= abs(x[2])` as new constraints. 

        ```ampl
        var U {1..2} >=0 <=100;
        var x {1..2} >=-30 <=100;
        minimize Obj: U[1] - 2*U[2];
        s.t. Con1:  3*x[1] - 2*x[2] <=  8;
        s.t. con2:    x[1] +   x[2] == 14;
        s.t. U1:    U[1] >= abs(x[1]);
        s.t. U2:    U[2] <= abs(x[2]);
        ```

        We can use the first (easy) reformulation above for `U[1]` and the second reformulation for `U[2]`.

        ```ampl
        var U {1..2} >=0 <=100;
        var b binary;
        s.t. AbsU1:
            U[1] >= x[1] and U[1] >= -x[1];
        s.t. AbsU2: 
            (b ==> (U[2] <= x[2])) and ((not b) ==> (U[2] <= -x[2]));
        ```

        If the solver requires, `AbsU2` can be linearized with big-M (assuming finite bounds on `x[2]`):

        ```ampl
        s.t. LinAbsU2:
            U[2] - x[2] + 30*b <= 30 and U[2] + x[2] - 200*b <= 0;
        ```

        Complete linearized model:

        ```ampl
        var x {1..2} >=-30 <=100;
        var U {1..2} >=0 <=100;
        var b binary;
        minimize Obj: U[1] - 2*U[2];
        s.t. Con1:  3*x[1] - 2*x[2] <=  8;
        s.t. Con2:    x[1] +   x[2] == 14;
        s.t. AbsU1: 
            U[1] >= x[1] and U[1] >= -x[1];
        s.t. LinAbsU2:
            U[2] - x[2] + 30*b <= 30 and U[2] + x[2] - 200*b <= 0;
        ```
        """
    )
