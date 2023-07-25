import streamlit as st
from .tip7_utils import LOGISTIC_REGRESSION_MOD, LOGISTIC_REGRESSION_CONIC_MOD
from .tip7_utils import run_small_dataset
import inspect

title = "Tip #7: Logistic Regression"


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
    ampl.param["lambd"] = lambd

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
    ampl.param["lambd"] = lambd

    # solve
    ampl.option["solver"] = solver
    ampl.solve()
    assert ampl.get_value("solve_result") == "solved"

    # return solution
    return ampl.var["theta"].to_pandas()


def run():
    st.markdown(
        r"""
        #### 1. Convex optimization model for logistic regression

        Define the logistic function $$ S(x)=\frac{1}{1+e^{-x}}.$$

        Next, given an observation $x\in\mathbf{R}^d$ and weights $\theta\in\mathbf{R}^d$ we set $$ h_\theta(x)=S(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}.$$

        The weights vector $\theta$ is part of the setup of the classifier. The expression
        $h_\theta(x)$ is interpreted as the probability that $x$ belongs to class 1.
        When asked to classify $x$ the returned answer is

        $$
        \begin{split}x\mapsto \begin{cases}\begin{array}{ll}1, & h_\theta(x)\geq 1/2, \\ 0, & h_\theta(x) < 1/2.\end{array}\end{cases}\end{split}
        $$

        When training a logistic regression algorithm we are given a sequence of training examples $x_i$, each labelled with its class $y_i\in \{0,1\}$ and we seek to find the weights
        $\theta$ which maximize the likelihood function $\textstyle \prod_i h_\theta(x_i)^{y_i}(1-h_\theta(x_i))^{1-y_i}$.
        Of course every single $y_i$ equals 0 or 1, so just one factor appears in the product for each training data point:

        $$\hspace{5em} \max_\theta \textstyle \prod_{i:y_i=1} h_\theta(x_i) \prod_{i:y_i=0} (1-h_\theta(x_i).$$

        By taking logarithms we obtain a sum that is easier to optimize:

        $$\hspace{5em} \max_\theta \sum_{i:y_i=1} \log(h_\theta(x_i))+\sum_{i:y_i=0}\log(1-h_\theta(x_i)). $$

        Note that by negating we obtain the logarithmic loss function:

        $$\hspace{5em} J(\theta) = -\sum_{i:y_i=1} \log(h_\theta(x_i))-\sum_{i:y_i=0}\log(1-h_\theta(x_i)). $$

        The training problem with regularization (a standard technique to prevent overfitting) is now equivalent to

        $$\hspace{5em} \min_\theta J(\theta) + \lambda\|\theta\|_2. $$


        This formulation can be solved with a general nonlinear solver, such as Ipopt.
        """
    )

    st.markdown("Convex optimization model for Logistic Regression:")

    st.code(
        LOGISTIC_REGRESSION_MOD,
        language="python",
        line_numbers=True,
    )

    st.markdown("Using from Python with amplpy:")

    st.code(
        inspect.getsource(logistic_regression),
        language="python",
        line_numbers=True,
    )

    st.markdown(
        r"""
        #### 2. Conic optimization model for logistic regression

        For a conic solver such as Mosek, we need to reformulate the problem.

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

    st.markdown("Conic optimization model for Logistic Regression:")

    st.code(
        LOGISTIC_REGRESSION_CONIC_MOD,
        language="python",
        line_numbers=True,
    )

    st.markdown("Using from Python with amplpy:")

    st.code(
        inspect.getsource(logistic_regression_conic),
        language="python",
        line_numbers=True,
    )

    run_small_dataset()

    st.markdown(
        """
        #### [[Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tips)]
        """
    )
