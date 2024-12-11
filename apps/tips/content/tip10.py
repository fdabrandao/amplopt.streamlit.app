import streamlit as st
from .utils import snippet

title = "Tip #10: Conditional disequality `b <==> x!=y`"


def run():
    st.markdown(
        """
        ### In AMPL you can simply write `b <==> x!=y`. How is conditional disequality linearized?

        ```python
        switch_state[t] = 1 <==> generator_state[t] != generator_state[prev(t)]
        ```

        *   This is equivalent to `b <== (x!=y)` and `b ==> (x!=y)`  
            which have very different linearizations!
            
        *   **AMPL MP reformulates this automatically for you** according to what each solver supports!
            
            *   For `b <== (x!=y)` we need the disjunction  
                `(b) or (x==y)`,  
                or, equivalently, the indicator constraint  
                `(not b) ==> (x==y)`.
                
            * For `b ==> (x!=y)` we need the disjunction  
            `(not b) or (eps <= x-y) or (eps <= -x+y)`  
            where `eps` is at least 1 for `x`, `y` integer,
            and the value of option _cvt:mip:eps_ otherwise.
            Note that no tolerance constant was required
            for the previous case (easy!)

            * The disjunctions may need to be linearized.
        """
        # TODO: expand example
    )
