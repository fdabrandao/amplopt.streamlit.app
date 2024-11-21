import streamlit as st
from .utils import snippet

title = "Tip #9: Variable disequality `x!=y`"


def run():
    st.markdown(
        """
        ### In AMPL you can simply write `x!=y`. How is this disequality linearized?

        *   Simply as `abs(x - y) >= eps`  
            where `eps` is at least 1 if `x` and `y` are integer.
            
        *   Which from **Modeling Tip #8** is linearized as:
            
            *   For `eps <= abs(x-y)` we need the disjunctive constraint  
                `(eps <= x-y) or (eps <= -x+y)`  
                which can be further reformulated as follows:  
                `B1 ==> (eps <= x-y)`  
                `B2 ==> (eps <= -x+y)`  
                `B1 + B2 >= 1`  
                where indicators can be linearized using Big-M.

        ## Example

        ```python
        s.t. DiffLanes {(i, j) in DiffLanePairs}:
            in_lane_veh[i] != in_lane_veh[j];
        ```
        """
    )
