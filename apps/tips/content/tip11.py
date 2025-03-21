import streamlit as st
from .utils import snippet

title = "Tip #11: Conditional absolute `b ==> abs(x) <= u`"


def run():
    st.markdown(
        """
        ### In AMPL you can simply write `b ==> abs(x) <= u`. How is this conditional absolute linearized?

        Consider constraints limiting changes of charge rate for a battery.

        ```python
        set T ordered;
        param MaxChargeRate;
        var ChargeRate{T} >= 0 <= MaxChargeRate;
        s.t. ChargeRateVariationLimit{t in T: ord(t) >= 2}:
            ChargeRate[t] != 0 && ChargeRate[prev(t)] != 0
                ==> 
            abs(ChargeRate[t]-ChargeRate[prev(t)]) <= 10;
        ```
        Note that we have the ‘easy’ case from [Tip #10](/Modeling_Tips?tip=10) here: `ChargeRate[t] != 0 ==> ...`.  
        The constraints can be reformulated as follows:
        ```python
        s.t. ChargeRateVariationLimitOR{t in T: ord(t) >= 2}:
            ChargeRate[t] == 0 or ChargeRate[prev(t)] == 0
            or abs(ChargeRate[t]-ChargeRate[prev(t)]) <= 10;
        ```
        To translate this for most solvers, we need to further
        reformulate using indicator constraints:
        ```python
        var ChargeRateZero {t in T} binary;
        var ChargeRateChangeLimited {t in T: ord(t) >= 2} binary;
        s.t. ChargeRateZeroIND{t in T}:
            ChargeRateZero[t] ==> ChargeRate[t] == 0;
        s.t. ChargeRateVariationLimitIND{t in T: ord(t) >= 2}:
            ChargeRateChangeLimited[t]
                ==> abs(ChargeRate[t]-ChargeRate[prev(t)]) <= 10;
        s.t. ChargeRateVariationLimitOR_IND{t in T: ord(t) >= 2}:
            ChargeRateZero[t] or ChargeRateZero[prev(t)]
            or ChargeRateChangeLimited[t];
        ```

        Finally, for solvers requiring full linearization, we use big-M
        (for `abs()` see **AMPL Modeling Tip #8: How is `y = abs(x)` linearized?**)

        - `a or b or c` can be linearized as `a+b+c >= 1`
        - `a ==> x <= u` with `x <= ub_x` can be linearized as `x <= ub_x - (ub_x-u)*a`

        ```python
        s.t. ChargeRateZeroIND_LIN{t in T}:
            ChargeRate[t] <= MaxChargeRate - MaxChargeRate*ChargeRateZero[t];
        s.t. ChargeRateVariationLimitIND_LIN{t in T: ord(t) >= 2}:
            ChargeRate[t]-ChargeRate[prev(t)] <= 
                    MaxChargeRate - (MaxChargeRate - 10) * ChargeRateChangeLimited[t]
            and
            ChargeRate[prev(t)]-ChargeRate[t] <= 
                    MaxChargeRate - (MaxChargeRate - 10) * ChargeRateChangeLimited[t];
        s.t. ChargeRateVariationLimitOR_IND_LIN{t in T: ord(t) >= 2}:
            ChargeRateZero[t] + ChargeRateZero[prev(t)]
            + ChargeRateChangeLimited[t] >= 1;
        ```

        Complete linearized example:
        ```python
        set T ordered;
        param MaxChargeRate;
        var ChargeRate{T} >= 0 <= MaxChargeRate;
        var ChargeRateZero {t in T} binary;
        var ChargeRateChangeLimited {t in T: ord(t) >= 2} binary;

        s.t. ChargeRateZeroIND_LIN{t in T}:
            ChargeRate[t] + MaxChargeRate*ChargeRateZero[t] <= MaxChargeRate;
        s.t. ChargeRateVariationLimitIND_LIN{t in T: ord(t) >= 2}:
            (MaxChargeRate - 10) * ChargeRateChangeLimited[t]
                + ChargeRate[t]-ChargeRate[prev(t)] <= MaxChargeRate
            and
            (MaxChargeRate - 10) * ChargeRateChangeLimited[t]
                + ChargeRate[prev(t)]-ChargeRate[t] <= MaxChargeRate;
        s.t. ChargeRateVariationLimitOR_IND_LIN{t in T: ord(t) >= 2}:
            ChargeRateZero[t] + ChargeRateZero[prev(t)] + ChargeRateChangeLimited[t] >= 1;
        ```
        """
    )
