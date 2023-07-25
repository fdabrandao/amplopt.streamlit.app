import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

FILTER_DATA_FOR_MOSEK = True
SMALL_DATASET = "https://raw.githubusercontent.com/ampl/amplcolab/master/datasets/regression/logistic_regression_ex2data2.csv"

DEGREE_LIFT = 6
DEGREE_STEP = 1

LOGISTIC_REGRESSION_MOD = """
set POINTS;
set DIMS;                  # Dimensionality of x

param y{POINTS} binary;    # Points' classes
param x{POINTS, DIMS};
param lambd;               # Regularization parameter

var theta{DIMS};           # Regression parameter vector
var hTheta{i in POINTS}
    = 1 / (1 + exp( - sum{d in DIMS} theta[d]*x[i, d] ));

minimize Logit:            # General nonlinear formulation
    - sum {i in POINTS: y[i] >0.5} log( hTheta[i] )
    - sum {i in POINTS: y[i]<=0.5} log( 1.0 - hTheta[i] )
    + lambd * sqrt( sum {d in DIMS} theta[d]^2 );
"""


LOGISTIC_REGRESSION_CONIC_MOD = """
set POINTS;
set DIMS;                  # Dimensionality of x

param y{POINTS} binary;    # Points' classes
param x{POINTS, DIMS};
param lambd;               # Regularization parameter

var theta{DIMS};           # Regression parameter vector
var t{POINTS};
var u{POINTS};
var v{POINTS};
var r >= 0;

minimize LogitConic:
    sum {i in POINTS} t[i] + lambd * r;

s.t. Softplus1{i in POINTS}:  # reformulation of softplus
    u[i] + v[i] <= 1;
s.t. Softplus2{i in POINTS}:
    u[i] >= exp(
        (if y[i]>0.5 then     # y[i]==1
            -sum {d in DIMS} theta[d] * x[i, d]
        else
            sum {d in DIMS} theta[d] * x[i, d]
        ) - t[i]
    );
s.t. Softplus3{i in POINTS}:
    v[i] >= exp(-t[i]);

s.t. Norm_Theta:              # Quadratic cone for regularizer
    r^2 >= sum {d in DIMS} theta[d]^2;
"""


class LogisticRegression:
    def __init__(self, label, data, modfile):
        from amplpy import AMPL

        self.ampl = ampl = AMPL()
        if modfile == "logistic_regression.mod":
            ampl.eval(LOGISTIC_REGRESSION_MOD)
        elif modfile == "logistic_regression_conic.mod":
            ampl.eval(LOGISTIC_REGRESSION_CONIC_MOD)
        else:
            assert False

        # load the data
        ampl.set["POINTS"] = data.index
        ampl.set["DIMS"] = data.columns
        ampl.param["y"] = label
        ampl.param["x"] = data

    def solve(self, solver, lambd):
        ampl = self.ampl
        ampl.param["lambd"] = lambd
        # solve
        ampl.option["solver"] = solver  # mosek, ipopt, knitro
        ampl.eval("let{d in DIMS} theta[d] := 0.0001;")  # initial guesses for IPOPT
        tm = time.perf_counter()
        solve_output = ampl.get_output("solve;")
        tm = time.perf_counter() - tm

        solve_result = ampl.get_value("solve_result")
        solve_message = ampl.get_value("solve_message")
        print(solve_message.strip())
        if solve_result != "solved":
            print(f"Warning: solve_result = {solve_result}")
            # print(solve_output.strip())

        # return solution and solve time
        return ampl.get_variable("theta").to_pandas(), tm


def logistic_regression(label, data, lambd, modfile, solver):
    from amplpy import AMPL

    # Create AMPL instance and load the model
    ampl = AMPL()
    if modfile == "logistic_regression.mod":
        ampl.eval(LOGISTIC_REGRESSION_MOD)
    elif modfile == "logistic_regression_conic.mod":
        ampl.eval(LOGISTIC_REGRESSION_CONIC_MOD)
    else:
        assert False

    # load the data
    ampl.set["POINTS"] = data.index
    ampl.set["DIMS"] = data.columns
    ampl.param["y"] = label
    ampl.param["x"] = data
    ampl.param["lambd"] = lambd

    # solve
    ampl.option["solver"] = solver  # mosek, ipopt, knitro
    ampl.eval("let{d in DIMS} theta[d] := 0.0001;")  # initial guesses for IPOPT
    tm = time.perf_counter()
    solve_output = ampl.get_output("solve;")
    tm = time.perf_counter() - tm

    solve_result = ampl.get_value("solve_result")
    solve_message = ampl.get_value("solve_message")
    print(solve_message.strip())
    if solve_result != "solved":
        print(f"Warning: solve_result = {solve_result}")
        # print(solve_output.strip())

    # return solution and solve time
    return ampl.get_variable("theta").to_pandas(), tm


def split_df(df, demo_limits=False):
    sample_size = int(df.shape[0] * 0.70)
    if demo_limits:  # adjust sample size to work under demo limit for MOSEK
        sample_size = min(sample_size, int((500 - 28) / 7))
    train_df = df.sample(n=sample_size, random_state=123)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def safe_pow(x, p):
    if np.min(x) > 0 or p.is_integer():
        return x**p
    x1 = np.array(x)
    x2 = np.array(x)
    x1[x1 < 0] = -((-x1[x1 < 0]) ** p)
    x2[x2 > 0] = x2[x2 > 0] ** p
    x2[x < 0] = x1[x < 0]
    return x2


def lift_to_degree(x, y, deg, step=1):
    result = pd.DataFrame()
    for i in np.arange(0, deg + step, step):
        for j in np.arange(0, i + step, step):
            temp = safe_pow(x, i) + safe_pow(y, (i - j))
            result[f"V{i}{i-j}"] = temp
    return result


def load_small_dataset():
    df = pd.read_csv(
        SMALL_DATASET,
        header=None,
        names=["Test1", "Test2", "Label"],
    )
    train_df, test_df = split_df(df, demo_limits=FILTER_DATA_FOR_MOSEK)
    train_df_lifted = lift_to_degree(
        train_df["Test1"], train_df["Test2"], DEGREE_LIFT, DEGREE_STEP
    )
    test_df_lifted = lift_to_degree(
        test_df["Test1"], test_df["Test2"], DEGREE_LIFT, DEGREE_STEP
    )
    return train_df, test_df, train_df_lifted, test_df_lifted


def plot_regression(qa, lambd, theta, ax):
    x, y, c = qa["Test1"], qa["Test2"], qa["Label"]
    ax.scatter(x, y, c=c, label="Good", alpha=0.3)
    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
    xd = (x1 - x0) / 10
    yd = (y1 - y0) / 10
    xr = np.linspace(x0 - xd, x1 + xd, 500)
    yr = np.linspace(y0 - xd, y1 + xd, 500)
    X, Y = np.meshgrid(xr, yr)  # grid of points
    X1 = np.reshape(X, (500 * 500))
    Y1 = np.reshape(Y, (500 * 500))
    X1Y1lft = lift_to_degree(X1, Y1, DEGREE_LIFT, DEGREE_STEP)
    theta_by_X1Y1 = theta.T @ X1Y1lft.T
    # Fails on Colab with Python 3.10:
    # theta_by_XY = np.reshape(theta_by_X1Y1, (500, 500))
    Z = np.zeros_like(X)
    for i in range(500):
        for j in range(500):
            Z[i, j] = 1 if -theta_by_X1Y1.iloc[0, i * 500 + j] > 0 else 0
    cp = ax.contour(X, Y, Z)
    ax.set_title(f"lambda = {lambd}")


def solve_and_subplot(train_df, train_df_lifted, lambd, ax, mdl, slv):
    theta, tm = logistic_regression(train_df["Label"], train_df_lifted, lambd, mdl, slv)
    print(f"Solving time: {tm:.2f} sec.")
    # display(theta.T)
    plot_regression(train_df, lambd, theta, ax)
    return theta, tm


def benchmark_lambda(train_df, train_df_lifted, modfile, solver):
    logistic = LogisticRegression(train_df["Label"], train_df_lifted, modfile)
    theta = {}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for lambd, axis in zip([0, 0.1, 1, 10], np.ravel(axes).tolist()):
        theta[lambd], tm = logistic.solve(solver, lambd)

        train_df["pred"] = train_df_lifted.dot(theta[lambd]) >= 0
        accuracy = sum(train_df["Label"] == train_df["pred"]) / len(train_df)
        st.markdown(
            f"- lambda={lambd:5.2f}:\n"
            f"\t- Solving time: {tm:.2f} seconds\n"
            f"\t- Accuracy on training data: {accuracy:5.2f}."
        )
        plot_regression(train_df, lambd, theta[lambd], axis)
    st.pyplot(plt)
    return theta


def evaluate_performance(theta, test_df, test_df_lifted):
    for lambd in sorted(theta):
        test_df["pred"] = test_df_lifted.dot(theta[lambd]) >= 0
        accuracy = sum(test_df["Label"] == test_df["pred"]) / len(test_df)
        st.markdown(
            f"- lambda={lambd:5.2f}:\n" f"\t- Accuracy on test data: {accuracy:5.2f}."
        )


def solve(train_df, test_df, train_df_lifted, test_df_lifted):
    st.markdown("## Solving logistic_regression.mod with IPOPT:")
    theta_logistic_ipopt = benchmark_lambda(
        train_df, train_df_lifted, "logistic_regression.mod", "ipopt"
    )
    evaluate_performance(theta_logistic_ipopt, test_df, test_df_lifted)

    st.markdown("## Solving logistic_regression_conic.mod with IPOPT:")
    theta_conic_ipopt = benchmark_lambda(
        train_df, train_df_lifted, "logistic_regression_conic.mod", "ipopt"
    )
    evaluate_performance(theta_conic_ipopt, test_df, test_df_lifted)

    st.markdown("## Solving logistic_regression_conic.mod with MOSEK:")
    theta_conic_mosek = benchmark_lambda(
        train_df, train_df_lifted, "logistic_regression_conic.mod", "mosek"
    )
    evaluate_performance(theta_conic_mosek, test_df, test_df_lifted)


def run_small_dataset():
    train_df, test_df, train_df_lifted, test_df_lifted = load_small_dataset()
    solve(train_df, test_df, train_df_lifted, test_df_lifted)
