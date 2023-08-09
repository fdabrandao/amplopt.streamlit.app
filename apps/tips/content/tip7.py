import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import time

SMALL_DATASET = "https://raw.githubusercontent.com/ampl/amplcolab/master/datasets/regression/logistic_regression_ex2data2.csv"
LARGER_DATASET = "https://raw.githubusercontent.com/ampl/amplcolab/master/datasets/regression/data_banknote_authentication.csv"

# limit sample size to work under demo limits for MOSEK
MAX_SAMPLE_SIZE = int((500 - 28) / 7)

DEGREE_LIFT = 6
DEGREE_STEP = 1

title = "Tip #7: Logistic Regression with Exponential Cones"

LOGISTIC_REGRESSION_MOD = """
set POINTS;
set DIMS;                  # Dimensionality of x

param y{POINTS} binary;    # Points' classes
param x{POINTS, DIMS};
param lambda;              # Regularization parameter

var theta{DIMS};           # Regression parameter vector
var hTheta{i in POINTS}
    = 1 / (1 + exp( - sum{d in DIMS} theta[d]*x[i, d] ));

minimize LogisticReg:      # General nonlinear formulation
    - sum {i in POINTS: y[i] >0.5} log( hTheta[i] )
    - sum {i in POINTS: y[i]<=0.5} log( 1.0 - hTheta[i] )
    + lambda * sqrt( sum {d in DIMS} theta[d]^2 );
"""


LOGISTIC_REGRESSION_CONIC_MOD = """
set POINTS;
set DIMS;                  # Dimensionality of x

param y{POINTS} binary;    # Points' classes
param x{POINTS, DIMS};
param lambda;              # Regularization parameter

var theta{DIMS};           # Regression parameter vector
var t{POINTS};
var u{POINTS};
var v{POINTS};
var r >= 0;

minimize LogisticRegConic:
    sum {i in POINTS} t[i] + lambda * r;

s.t. Softplus1{i in POINTS}:  # reformulation of softplus
    u[i] + v[i] <= 1;
s.t. Softplus2{i in POINTS}:
    u[i] >= exp(
        (if y[i]>0.5 then -1 else 1)
        * (sum {d in DIMS} theta[d] * x[i, d])
        - t[i]
    );
s.t. Softplus3{i in POINTS}:
    v[i] >= exp(-t[i]);

s.t. NormTheta:              # Quadratic cone for regularizer
    r^2 >= sum {d in DIMS} theta[d]^2;
"""


def logistic_regression(label, data, lambd, solver):
    from amplpy import AMPL

    # load the model
    ampl = AMPL()
    ampl.read("logistic_regression.mod")

    # load the data
    ampl.set["POINTS"] = data.index
    ampl.set["DIMS"] = data.columns
    ampl.param["x"] = data
    ampl.param["y"] = label
    ampl.param["lambda"] = lambd

    # solve
    ampl.option["solver"] = solver
    ampl.solve()
    assert ampl.get_value("solve_result") == "solved"

    # return solution
    return ampl.var["theta"].to_pandas()


def logistic_regression_conic(label, data, lambd, solver):
    from amplpy import AMPL

    # load the model
    ampl = AMPL()
    ampl.read("logistic_regression_conic.mod")

    # load the data
    ampl.set["POINTS"] = data.index
    ampl.set["DIMS"] = data.columns
    ampl.param["x"] = data
    ampl.param["y"] = label
    ampl.param["lambda"] = lambd

    # solve
    ampl.option["solver"] = solver
    ampl.solve()
    assert ampl.get_value("solve_result") == "solved"

    # return solution
    return ampl.var["theta"].to_pandas()


def header():
    st.markdown(
        r"""
        Given a sequence of training examples $x_i \in \mathbf{R}^m$, each labelled with its class $y_i\in \{0,1\}$ and we seek to find the weights $\theta \in \mathbf{R}^m$ which maximize the function:

        $$
        \sum_{i:y_i=1} \log(S(\theta^Tx_i))+\sum_{i:y_i=0}\log(1-S(\theta^Tx_i))
        $$

        where $S$ is the logistic function $S(x) = \frac{1}{1+e^{-x}}$ that estimates the probability of a binary classifier to be 0 or 1.

        **This function can be efficiently optimized using exponential cones with [MOSEK](https://ampl.com/products/solvers/solvers-we-sell/mosek/)!**
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image("static/apps/tips/LogisticRegression.png")
    with col2:
        st.image("static/apps/tips/Sigmoid.png")

    st.markdown(
        r"""
        ## 1.1. Convex optimization model for logistic regression

        Define the logistic function
        $$
        S(x)=\frac{1}{1+e^{-x}}.
        $$

        Next, given an observation $x\in\mathbf{R}^d$ and weights $\theta\in\mathbf{R}^d$ we set
        $$
        h_\theta(x)=S(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}.
        $$

        The weights vector $\theta$ is part of the setup of the classifier. The expression

        $h_\theta(x)$ is interpreted as the probability that $x$ belongs to class 1.

        When asked to classify $x$ the returned answer is

        $$
        x\mapsto \begin{cases}\begin{array}{ll}1, & h_\theta(x)\geq 1/2, \\ 0, & h_\theta(x) < 1/2.\end{array}\end{cases}
        $$

        When training a logistic regression algorithm we are given a sequence of training examples $x_i$, 
        each labelled with its class $y_i\in \{0,1\}$ and we seek to find the weights $\theta$ which 
        maximize the likelihood function $\textstyle \prod_i h_\theta(x_i)^{y_i}(1-h_\theta(x_i))^{1-y_i}$.

        Of course every single $y_i$ equals 0 or 1, so just one factor appears in the product for each training data point:

        $$
        \max_\theta \textstyle \prod_{i:y_i=1} h_\theta(x_i) \prod_{i:y_i=0} (1-h_\theta(x_i)).
        $$

        By taking logarithms we obtain a sum that is easier to optimize:

        $$
        \max_\theta \sum_{i:y_i=1} \log(h_\theta(x_i))+\sum_{i:y_i=0}\log(1-h_\theta(x_i)).
        $$

        Note that by negating we obtain the logarithmic loss function:

        $$
        J(\theta) = -\sum_{i:y_i=1} \log(h_\theta(x_i))-\sum_{i:y_i=0}\log(1-h_\theta(x_i)).
        $$

        The training problem with regularization (a standard technique to prevent overfitting) is now equivalent to

        $$
        \min_\theta J(\theta) + \lambda\|\theta\|_2.
        $$

        This formulation can be solved with a general nonlinear solver, such as [IPOPT](https://ampl.com/products/solvers/open-source-solvers/).
        """
    )

    st.markdown(
        """
        ### Convex optimization model for Logistic Regression

        Model file "**logistic_regression.mod**":
        """
    )

    st.code(
        LOGISTIC_REGRESSION_MOD,
        language="python",
        line_numbers=True,
    )

    st.markdown("Using from Python with [amplpy](https://amplpy.readthedocs.io):")

    st.code(
        inspect.getsource(logistic_regression),
        language="python",
        line_numbers=True,
    )

    st.markdown(
        r"""
        ## 1.2. Conic optimization model for logistic regression

        For a conic solver such as [MOSEK](https://ampl.com/products/solvers/solvers-we-sell/mosek/), we need to reformulate the problem.

        The objective function can equivalently be phrased as

        $$
        \begin{split}\begin{array}{lrllr}
        \min & \sum_i t_i +\lambda r  \\
        \text{s.t.} & t_i      & \geq - \log(h_\theta(x))   & = \log(1+e^{-\theta^Tx_i}) & \mathrm{if}\ y_i=1, \\
                & t_i      & \geq - \log(1-h_\theta(x)) & = \log(1+e^{\theta^Tx_i})  & \mathrm{if}\ y_i=0, \\
                & r        & \geq \|\theta\|_2.
        \end{array}\end{split}
        $$

        The key point is to implement the [softplus bound](https://docs.mosek.com/modeling-cookbook/expo.html#softplus-function) $t\geq \log(1+e^u)$, which is the simplest example of a log-sum-exp constraint for two terms. Here $t$ is a scalar variable and $u$ will be the affine expression of the form $\pm \theta^Tx_i$. This is equivalent to

        $$
        e^{u-t} + e^{-t} \leq 1
        $$

        and further to

        $$
        \begin{split}\begin{array}{rclr}
        z_1  & \ge  & e^{u-t}, \\
        z_2  & \ge  & e^{-t}, \\
        z_1+z_2       & \leq & 1.
        \end{array}\end{split}
        $$
    """
    )

    st.markdown(
        """
        ### Conic optimization model for Logistic Regression

        Model file "**logistic_regression_conic.mod**":
        """
    )

    st.code(
        LOGISTIC_REGRESSION_CONIC_MOD,
        language="python",
        line_numbers=True,
    )

    st.markdown("Using from Python with [amplpy](https://amplpy.readthedocs.io):")

    st.code(
        inspect.getsource(logistic_regression_conic),
        language="python",
        line_numbers=True,
    )


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
        ampl.param["lambda"] = lambd
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
            result[f"V{i}{i-j}"] = safe_pow(x, i) + safe_pow(y, (i - j))
    return result


class InputDataset:
    def __init__(self, df: pd.DataFrame, label: str):
        assert list(df.columns) == ["Feature1", "Feature2", "Label"]
        # split
        train_df, test_df = self._split_df(df)

        # lift
        train_df_lifted = lift_to_degree(
            train_df["Feature1"], train_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
        )
        test_df_lifted = lift_to_degree(
            test_df["Feature1"], test_df["Feature2"], DEGREE_LIFT, DEGREE_STEP
        )

        self.train_df, self.train_df_lifted = train_df, train_df_lifted
        self.test_df, self.test_df_lifted = test_df, test_df_lifted
        self.label = label
        self.df = df

    def _split_df(self, df):
        sample_size = int(df.shape[0] * 0.70)
        if MAX_SAMPLE_SIZE is not None:
            sample_size = min(sample_size, MAX_SAMPLE_SIZE)
        train_df = df.sample(n=sample_size, random_state=123)
        test_df = df.drop(train_df.index)
        return train_df, test_df

    def plot(self):
        _, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(
            x=self.df["Feature1"],
            y=self.df["Feature2"],
            c=self.df["Label"],
            label=self.label,
            alpha=0.5,
        )
        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        ax.legend()
        st.pyplot(plt)


def load_small_dataset():
    data = pd.read_csv(
        SMALL_DATASET,
        header=None,
        names=["Feature1", "Feature2", "Label"],
    )
    return data, InputDataset(data, "Good")


def load_larger_dataset():
    data = pd.read_csv(
        LARGER_DATASET,
        names=["variance", "skewness", "curtosis", "entropy", "class"],
        header=None,
    )
    df = data[["variance", "skewness", "class"]].copy()
    df.columns = ["Feature1", "Feature2", "Label"]

    # normalize
    df["Feature1"] /= df["Feature1"].abs().max()
    df["Feature2"] /= df["Feature2"].abs().max()
    return data, InputDataset(df, "Counterfeit")


class ModelEvaluator:
    def __init__(self, dataset: InputDataset, model: str):
        self.dataset = dataset
        self.model = model
        self.classifier = LogisticRegression(
            dataset.train_df["Label"], dataset.train_df_lifted, model
        )

    def test(self, solver: str):
        st.markdown(f"### {self.model} with {solver.upper()}:")
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
                f"- lambda = {lambd:.2f}:\n"
                f"\t- Solving time: {tm[lambd]:.2f} seconds.\n"
                f"\t- Accuracy on training data: {train_accuracy:5.2f}.\n"
                f"\t- Accuracy on testing data: {test_accuracy:5.2f}.\n"
            )

        st.markdown("- Visual split of training data:")
        _, axes = plt.subplots(2, 2, figsize=(9, 7))
        for lambd, ax in zip(lambda_values, np.ravel(axes).tolist()):
            self._plot_regression(ds.train_df, lambd, theta[lambd], ax)
        st.pyplot(plt)

        st.markdown("- Visual split of testing data:")
        _, axes = plt.subplots(2, 2, figsize=(9, 7))
        for lambd, ax in zip(lambda_values, np.ravel(axes).tolist()):
            self._plot_regression(ds.test_df, lambd, theta[lambd], ax)
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

        _, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        self._plot_regression(
            ds.train_df,
            lambd,
            theta[lambd],
            ax[0],
            "Visual split of training set",
        )
        self._plot_regression(
            ds.test_df,
            lambd,
            theta[lambd],
            ax[1],
            "Visual split of testing set",
        )
        st.pyplot(plt)

        st.markdown(
            f"\t- Solving time: {tm[lambd]:.2f} seconds.\n"
            f"\t- Accuracy on training data: {train_accuracy:.2f}.\n"
            f"\t- Accuracy on testing data: {test_accuracy:.2f}.\n"
        )

    def _plot_regression(self, dataset, lambd, theta, ax, title=None):
        x, y, c = dataset["Feature1"], dataset["Feature2"], dataset["Label"]
        ax.scatter(
            x,
            y,
            c=c,
            label=self.dataset.label,
            alpha=0.5,
        )
        ax.legend()
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


@st.cache_data
def classify_small_dataset():
    data, dataset = load_small_dataset()
    st.markdown(
        """
    ## 2. Small dataset

    In the first part, we will implement regularized logistic regression to predict
    whether microchips from a fabrication plant pass quality assurance (QA). During QA,
    each microchip goes through various tests to ensure it is functioning correctly.
    Suppose you are the product manager of the factory and you have the test results
    for some microchips on two different tests. From these two tests, you would like
    to determine whether the microchips should be accepted or rejected.
    """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(data)
    with col2:
        dataset.plot()
    st.markdown(
        """
    Logistic regression is an example of a binary classifier, where the output takes one of the two values 0 or 1 for each data point. We call the two values *classes*.

    **As we see from the plot, a linear separation of the classes is not reasonable. We lift the 2D data into $\mathbf{R}^{28}$ via sums of monomials of degrees up to 6.** [[See Google Colab Notebook for more details](https://colab.ampl.com/notebooks.html#logistic-regression-with-amplpy)]
    """
    )
    ModelEvaluator(dataset, "logistic_regression.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("mosek")


@st.cache_data
def classify_larger_dataset():
    data, dataset = load_larger_dataset()
    st.markdown(
        """
    ## 3. Larger dataset

    The second data set contains data from a collection of known genuine and
    known counterfeit banknote specimens. The data includes four continuous
    statistical measures obtained from the wavelet transform of banknote images
    named "variance", "skewness", "curtosis", and "entropy", and a binary variable
    named "class" which is 0 if genuine and 1 if counterfeit.
    """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(data)
    with col2:
        dataset.plot()
    st.markdown(
        """
    **From the 4 features we select 2 ("variance" and "skewness") to be able to
    visualize the results. Similar to the small example, we lift the 2D data into
    $\mathbf{R}^{28}$ via sums of monomials of degrees up to 6.** [[See Google Colab Notebook for more details](https://colab.ampl.com/notebooks.html#logistic-regression-with-amplpy)]
    """
    )
    ModelEvaluator(dataset, "logistic_regression.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("ipopt")
    ModelEvaluator(dataset, "logistic_regression_conic.mod").test("mosek")


def experiments():
    classify_small_dataset()
    classify_larger_dataset()

    st.markdown("# 4. Run experiments!")

    datasets = ["Small dataset", "Larger dataset"]
    dataset_name = st.selectbox("Pick the dataset 👇", datasets, key="dataset", index=1)
    if dataset_name == datasets[0]:
        _, dataset = load_small_dataset()
    else:
        _, dataset = load_larger_dataset()

    models = ["logistic_regression_conic.mod", "logistic_regression.mod"]
    model = st.selectbox("Pick the model 👇", models, key="model")

    evaluator = ModelEvaluator(dataset, model)

    solvers = ["mosek", "ipopt"] if "conic" in model else ["ipopt"]
    solver = st.selectbox("Pick the solver 👇", solvers, key="solver")

    lambd = st.slider(
        "Lambda multiplier to control overfitting with regularization 👇",
        0.0,
        5.0,
        1.0,
        step=0.01,
    )
    evaluator.test_lambda(solver, lambd)


def footer():
    st.markdown(
        """
    ## References:
    1. Small data set example:
        - https://docs.mosek.com/modeling-cookbook/expo.html#logistic-regression,
        - https://docs.mosek.com/latest/pythonapi/case-studies-logistic.html#.
    2. Large data set:
        - Lohweg, Volker. (2013). banknote authentication. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57.
    """
    )

    st.markdown(
        """
        #### [[Google Colab Notebook](https://colab.ampl.com/notebooks.html#logistic-regression-with-amplpy)] [[On LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7092187981612023808)] [[On Twitter](https://twitter.com/AMPLopt/status/1686424007422234624)] [[On Discourse](https://discuss.ampl.com/t/ampl-modeling-tips-7-logistic-regression-with-exponential-cones/657)] [[Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tips)]
        """
    )


def run():
    header()
    experiments()
    footer()
