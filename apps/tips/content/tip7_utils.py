import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import namedtuple
import time

FILTER_DATA_FOR_MOSEK = True
SMALL_DATASET = "https://raw.githubusercontent.com/ampl/amplcolab/master/datasets/regression/logistic_regression_ex2data2.csv"
LARGER_DATASET = "https://raw.githubusercontent.com/ampl/amplcolab/master/datasets/regression/data_banknote_authentication.csv"

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

    def optimize(self, solver, lambd):
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
        return ampl.var["theta"].to_pandas(), tm


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


Dataset = namedtuple(
    "Dataset", ["train_df", "train_df_lifted", "test_df", "test_df_lifted"]
)


def load_small_dataset():
    df = pd.read_csv(
        SMALL_DATASET,
        header=None,
        names=["Feature1", "Feature2", "Label"],
    )

    # split
    train_df, test_df = split_df(df, demo_limits=FILTER_DATA_FOR_MOSEK)

    # lift
    train_df_lifted = lift_to_degree(
        train_df["Feature1"], train_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
    )
    test_df_lifted = lift_to_degree(
        test_df["Feature1"], test_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
    )
    return Dataset(train_df, train_df_lifted, test_df, test_df_lifted)


def load_larger_dataset():
    df = pd.read_csv(
        LARGER_DATASET,
        names=["variance", "skewness", "curtosis", "entropy", "class"],
        header=None,
    )
    df = df[["variance", "skewness", "class"]]
    df.columns = ["Feature1", "Feature2", "Label"]

    # normalize
    df["Feature1"] /= df["Feature1"].abs().max()
    df["Feature2"] /= df["Feature2"].abs().max()

    # split
    train_df, test_df = split_df(df, demo_limits=FILTER_DATA_FOR_MOSEK)

    # lift
    train_df_lifted = lift_to_degree(
        train_df["Feature1"], train_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
    )
    test_df_lifted = lift_to_degree(
        test_df["Feature1"], test_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
    )
    return Dataset(train_df, train_df_lifted, test_df, test_df_lifted)


def plot_regression(dataset, lambd, theta, ax, title=None):
    x, y, c = dataset["Feature1"], dataset["Feature2"], dataset["Label"]
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
    Z = (theta_by_X1Y1.values > 0).astype(int).reshape(500, 500)
    ax.contour(X, Y, Z)
    if title is None:
        title = f"lambda = {lambd}"
    ax.set_title(title)


class ModelEvaluator:
    def __init__(self, dataset: Dataset, model: str):
        self.dataset = dataset
        self.model = model
        self.classifier = LogisticRegression(
            dataset.train_df["Label"], dataset.train_df_lifted, model
        )

    def test(self, solver: str):
        st.markdown(f"## {self.model} with {solver.upper()}:")
        ds = self.dataset
        theta = {}
        tm = {}
        lambda_values = [0, 0.1, 1, 10]
        for lambd in lambda_values:
            # Optimize
            theta[lambd], tm[lambd] = self.classifier.optimize(solver, lambd)

            # Training accuracy
            ds.train_df["pred"] = ds.train_df_lifted.dot(theta[lambd]) >= 0
            train_accuracy = sum(ds.train_df["Label"] == ds.train_df["pred"]) / len(
                ds.train_df
            )

            # Testing accuracy
            ds.test_df["pred"] = ds.test_df_lifted.dot(theta[lambd]) >= 0
            test_accuracy = sum(ds.test_df["Label"] == ds.test_df["pred"]) / len(
                ds.test_df
            )

            st.markdown(
                f"- lambda={lambd:5.2f}:\n"
                f"\t- Solving time: {tm[lambd]:.2f} seconds.\n"
                f"\t- Accuracy on training data: {train_accuracy:5.2f}.\n"
                f"\t- Accuracy on testing data: {test_accuracy:5.2f}.\n"
            )

        st.markdown("### Visual classification of training data:")
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for lambd, axis in zip(lambda_values, np.ravel(axes).tolist()):
            plot_regression(ds.train_df, lambd, theta[lambd], axis)
        st.pyplot(plt)

        st.markdown("### Visual classification of testing data:")
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for lambd, axis in zip(lambda_values, np.ravel(axes).tolist()):
            plot_regression(ds.test_df, lambd, theta[lambd], axis)
        st.pyplot(plt)

    def test_lambda(self, solver: str, lambd: float):
        st.markdown(f"## {self.model} with {solver.upper()}:")
        ds = self.dataset
        theta = {}
        tm = {}
        # Optimize
        theta[lambd], tm[lambd] = self.classifier.optimize(solver, lambd)

        # Training accuracy
        ds.train_df["pred"] = ds.train_df_lifted.dot(theta[lambd]) >= 0
        train_accuracy = sum(ds.train_df["Label"] == ds.train_df["pred"]) / len(
            ds.train_df
        )

        # Testing accuracy
        ds.test_df["pred"] = ds.test_df_lifted.dot(theta[lambd]) >= 0
        test_accuracy = sum(ds.test_df["Label"] == ds.test_df["pred"]) / len(ds.test_df)

        st.markdown(
            f"- lambda={lambd:.2f}:\n"
            f"\t- Solving time: {tm[lambd]:.2f} seconds.\n"
            f"\t- Accuracy on training data: {train_accuracy:.2f}.\n"
            f"\t- Accuracy on testing data: {test_accuracy:.2f}.\n"
        )

        st.markdown("### Visual classification of the data:")
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        plot_regression(ds.train_df, lambd, theta[lambd], ax[0], "Training set")
        plot_regression(ds.test_df, lambd, theta[lambd], ax[1], "Testing set")
        st.pyplot(plt)


@st.cache_data
def classify_small_dataset():
    st.markdown(
        """
    ## 2. Small dataset

    In the first part, we will implement regularized logistic regression to predict
    whether microchips from a fabrication plant pass quality assurance (QA). During QA,
    each microchip goes through various tests to ensure it is functioning correctly.
    Suppose you are the product manager of the factory and you have the test results
    for some microchips on two different tests. From these two tests, you would like
    to determine whether the microchips should be accepted or rejected.

    Logistic regression is an example of a binary classifier, where the output takes one of the two values 0 or 1 for each data point. We call the two values *classes*.

    As we see from the plot, a linear separation of the classes is not reasonable. We lift the 2D data into $\mathbf{R}^{28}$ via sums of monomials of degrees up to 6.
    """
    )
    dataset = load_small_dataset()
    ModelEvaluator(dataset, "logistic_regression.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("mosek")


@st.cache_data
def classify_larger_dataset():
    st.markdown(
        """
    ## 3. Larger dataset

    The setcond data set contains data from a collection of known genuine and
    known counterfeit banknote specimens. The data includes four continuous
    statistical measures obtained from the wavelet transform of banknote images
    named "variance", "skewness", "curtosis", and "entropy", and a binary variable
    named "class" which is 0 if genuine and 1 if counterfeit.

    From the 4 features we select 2 ("variance" and "skewness") to be able to
    visualize the results. Similar to the small example, we lift the 2D data into
    $\mathbf{R}^{28}$ via sums of monomials of degrees up to 6.
    """
    )
    dataset = load_larger_dataset()
    ModelEvaluator(dataset, "logistic_regression.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("mosek")


def run_experiments():
    classify_small_dataset()
    classify_larger_dataset()

    st.markdown("# 4. Run custom experiments!")

    datasets = ["Small dataset", "Larger dataset"]
    if st.selectbox("Pick the dataset 👇", datasets, key="dataset") == datasets[0]:
        dataset = load_small_dataset()
    else:
        dataset = load_larger_dataset()

    models = ["logistic_regression_conic.mod", "logistic_regression.mod"]
    model = st.selectbox("Pick the model 👇", models, key="model")

    solvers = ["mosek", "ipopt"] if "conic" in model else ["ipopt"]
    solver = st.selectbox("Pick the solver 👇", solvers, key="solver")

    lambd = st.slider("Lambda?", 0.0, 10.0, 1.0, step=0.01)
    ModelEvaluator(dataset, model).test_lambda(solver, lambd)
