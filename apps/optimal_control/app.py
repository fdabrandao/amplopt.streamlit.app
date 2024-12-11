import streamlit as st
from amplpy import AMPL
import matplotlib.pyplot as plt
from ..common import solver_selector, MP_SOLVERS_LINKS

MODEL = r"""
# Sets
param steps;
set T ordered := 1..steps; # Discrete time steps

# Parameters
param g default 9.81; # Gravitational constant (m/s^2)
param Tmax;           # Maximum thrust (N)
param m0;             # Initial mass of the rocket (kg)
param mdot;           # Mass flow rate (kg/s)
param ve;             # Exhaust velocity (m/s)

param x0 default 0;   # Initial x-position (m)
param y0 default 0;   # Initial y-position (m)
param vx0 default 0;  # Initial x-velocity (m/s)
param vy0 default 0;  # Initial y-velocity (m/s)

param xf;             # Final x-position target (m)
param yf;             # Final y-position target (m)
param dt;             # Time step duration (s)

# Decision Variables
var x{T};             # x-position of the rocket at time t
var y{T};             # y-position of the rocket at time t
var vx{T};            # x-velocity of the rocket at time t
var vy{T};            # y-velocity of the rocket at time t
var m{T} >= 0;        # Mass of the rocket at time t
var Tmag{T} >= 0, <= Tmax; # Thrust magnitude at time t
var Tx{T};            # Thrust in x-direction at time t
var Ty{T};            # Thrust in y-direction at time t

# Objective Function: Minimize fuel consumption
minimize Fuel_Consumption:
    sum {t in T} Tmag[t] * dt / ve;

# Dynamics and Constraints
subject to Dynamics_x {t in T: t != last(T)}:
    x[t+1] = x[t] + vx[t] * dt;

subject to Dynamics_y {t in T: t != last(T)}:
    y[t+1] = y[t] + vy[t] * dt;

subject to Dynamics_vx {t in T: t != last(T)}:
    vx[t+1] = vx[t] + (Tx[t] / m[t]) * dt;

subject to Dynamics_vy {t in T: t != last(T)}:
    vy[t+1] = vy[t] + (Ty[t] / m[t] - g) * dt;

subject to Dynamics_mass {t in T: t != last(T)}:
    m[t+1] = m[t] - (Tmag[t] / ve) * dt;

# Thrust vector constraints
subject to Thrust_Vector {t in T}:
    Tmag[t] = sqrt(1e-6+Tx[t]^2 + Ty[t]^2);

# Initial conditions
subject to Initial_Conditions_1:
    x[first(T)] = x0;
subject to Initial_Conditions_2:
    y[first(T)] = y0;
subject to Initial_Conditions_3:
    vx[first(T)] = vx0;
subject to Initial_Conditions_4:
    vy[first(T)] = vy0;
subject to Initial_Conditions_5:
    m[first(T)] = m0;

# Terminal conditions
subject to Terminal_Conditions_1:
    x[last(T)] = xf;
subject to Terminal_Conditions_2:
    y[last(T)] = yf;
"""


def main():
    st.title("🎯 Optimal Control by ChatGPT")

    st.code(MODEL)

    ampl = AMPL()
    ampl.eval(MODEL)

    c1, c2, c3 = st.columns(3)
    with c1:
        ampl.param["steps"] = st.slider("How many steps?", 10, 20, 10)
    with c2:
        ampl.param["dt"] = st.slider("Time step duration in seconds?", 1, 60, 1)
    with c3:
        ampl.param["m0"] = st.slider("Initial rocket mass in kg?", 100, 1000, 500)

    with c1:
        ampl.param["Tmax"] = st.slider(
            "Maximum thrust in Newtons?", 10000, 30000, 20000
        )
    with c2:
        ampl.param["mdot"] = st.slider("Mass flow rate in kg/s?", 5, 20, 10)
    with c3:
        ampl.param["ve"] = st.slider("Exhaust velocity in m/s?", 2000, 4000, 3000)

    c1, c2 = st.columns(2)
    with c1:
        ampl.param["x0"] = st.slider("Initial x-position?", 0, 1000, 0)
    with c2:
        ampl.param["y0"] = st.slider("Initial y-position?", 0, 1000, 0)
    with c1:
        ampl.param["xf"] = st.slider("Final x-position?", 0, 10000, 10000)
    with c2:
        ampl.param["yf"] = st.slider("Final y-position?", 0, 10000, 5000)
    output = ampl.solve(solver="snopt", return_output=True)
    st.markdown(f"```\n{output}\n```")

    df = ampl.get_data("x", "y", "vx", "vy", "m", "Tmag").to_pandas()
    st.dataframe(df)

    # fig, ax = plt.subplots()
    # ax.plot(
    #     df["x"], df["y"], marker="o"
    # )  # Connect points with a line and add markers for clarity
    # ax.set_title("Line Graph")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # st.pyplot(fig)

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/optimal_control)] [[ChatGPT Session](https://chatgpt.com/share/673f7f67-a2ec-8011-9633-01a3570ef26f)]
    """
    )
