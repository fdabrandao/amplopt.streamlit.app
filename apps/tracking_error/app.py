import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup
from amplpy import AMPL
import matplotlib.pyplot as plt
import seaborn as sns
from ..common import solver_selector, MP_SOLVERS_LINKS

TRACKING_ERROR_MODEL = r"""
# === Sets ===
set ASSETS;  # Available assets

# === Parameters ===
param sigma{ASSETS, ASSETS};       # Covariance matrix
param w_prev{ASSETS} default 0;    # Previous portfolio weights
param w_bench{ASSETS};             # Benchmark weights
param min_weight default 0.01;     # Min weight if invested
param max_weight default 0.20;     # Max weight per asset
param n_assets default 10;         # Target number of holdings
param turnover_limit default 0.05; # Max allowed turnover

# === Decision Variables ===
var w{ASSETS} >= 0, <= max_weight; # Portfolio weights (long-only)

# === Objective ===
minimize TrackingErrorSquared:
    sum{i in ASSETS, j in ASSETS} 
        (w[i] - w_bench[i]) * sigma[i,j] * (w[j] - w_bench[j])
    suffix objpriority 2;

minimize MinVariance:
    sum{i in ASSETS, j in ASSETS}
        w[i] * sigma[i,j] * w[j]
    suffix objpriority 1;

# === Constraints ===
s.t. FullyInvested:
    sum{i in ASSETS} w[i] = 1;  # All capital invested

s.t. PositionCount:
    count {i in ASSETS} (w[i] != 0) = n_assets;  # Exact number of holdings

s.t. MinWeightOrZero{i in ASSETS}:
    w[i] = 0 or w[i] >= min_weight;  # Either 0 or above min weight

s.t. Turnover{if sum{i in ASSETS} w_prev[i] > 0}:
    sum{i in ASSETS} abs(w[i] - w_prev[i]) <= turnover_limit;  # Limit turnover
"""


@st.cache_data
def get_sp500_symbols():
    """Get current S&P 500 constituents from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table.set_index("Symbol")


@st.cache_data
def get_sp500_weights():
    """Scrape S&P 500 constituents and their weights from websites like SlickCharts"""
    url = "https://www.slickcharts.com/sp500"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "table table-hover table-borderless table-sm"})

    symbols = []
    weights = []
    for row in table.find_all("tr")[1:]:  # Skip header
        cols = row.find_all("td")
        symbols.append(cols[2].text.strip())
        weights.append(float(cols[3].text.replace(",", "").strip("%")) / 100)

    return pd.DataFrame({"Symbol": symbols, "Weight": weights}).set_index("Symbol")


@st.cache_data
def download_price_data(symbols, start_date, end_date):
    """Download historical adjusted close prices for given symbols"""
    data = yf.download(
        tickers=list(symbols),
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )
    return data["Close"].dropna(how="any", axis=1)


def plot_pie_chart(weights, title):
    """Plot a pie chart of portfolio weights with improved visuals."""

    # Filter out the tickers with weight less than or equal to 1e-4
    filtered_weights = weights[weights > 1e-4]
    filtered_labels = filtered_weights.index

    # Set color palette
    palette = sns.color_palette("Set3", len(filtered_weights))

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        filtered_weights,
        labels=filtered_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=palette,
        explode=[0.1]
        * len(filtered_weights),  # Slightly "explode" all slices for emphasis
        wedgeprops={"edgecolor": "black", "linewidth": 1, "linestyle": "solid"},
    )

    # Improve text labels
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    # Add title
    plt.title(title, fontsize=18, fontweight="bold")

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis("equal")

    # Show the plot
    st.pyplot(plt)


def plot_portfolio_comparison(split_date, portfolio_returns):
    """Plot comparison of portfolio returns."""
    # Combine into a single DataFrame for plotting
    combined = pd.DataFrame(portfolio_returns)

    # Seaborn style
    sns.set_theme(style="whitegrid")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add vertical line for slit date
    ax.axvline(x=split_date, color="red", linestyle="--", label="Train/Test Split")

    # Plot using seaborn
    sns.lineplot(data=combined, ax=ax)
    ax.set_title("Portfolio Performance Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")

    # Add legend
    ax.legend()

    # Show in Streamlit
    st.pyplot(fig)


def main():
    st.title("📈 Tracking Error Optimization")

    st.markdown(
        """
        ### Build Portfolios That Closely Follow Your Benchmark
 
        Tracking error optimization is a technique used in portfolio management to minimize the deviation of a portfolio’s returns from its benchmark. This deviation, called *tracking error*, is a measure of active risk and is critical for strategies aiming to closely track an index while allowing for some degree of active management.

        Tracking error can be measured in two ways:

        - **Backward-looking tracking error** uses historical return data to assess how much a portfolio has deviated from its benchmark in the past. It’s useful for performance evaluation but may not reflect future conditions.

        - **Forward-looking tracking error** is based on forecasts of return volatility and correlations. It estimates expected future deviation and is used in optimization models to construct portfolios within a desired risk range.

        Both perspectives are essential — backward-looking for assessment, forward-looking for decision-making.
        The optimization model we will use is a quadratic programming model that minimizes the tracking error while adhering to constraints such as maximum weight per asset, minimum weight if invested, and turnover limits. The model is formulated in AMPL and solved using a suitable solver. 
        """
    )

    st.write("## Tracking Error Model in AMPL")
    st.code(TRACKING_ERROR_MODEL)

    # Get S&P 500 weights
    sp500_weights = get_sp500_weights()
    st.write("## Current S&P 500 weights")
    st.dataframe(sp500_weights.T)

    # Get S&P 500 symbols
    sp500 = get_sp500_symbols()
    diff_assets = set(sp500.index) - set(sp500_weights.index)
    assert len(diff_assets) == 0, diff_assets

    # Add Weight column to sp500 dataframe
    sp500["Weight"] = sp500_weights["Weight"]
    columns = list(sp500.columns)
    columns.remove("Weight")
    columns.insert(1, "Weight")
    sp500 = sp500[columns]

    # Filter top 100
    sp500 = sp500.nlargest(100, "Weight")

    st.write("## Top #100 of current S&P 500 constituents")
    st.dataframe(sp500)

    # Training data starts at
    min_start_date = datetime.date(2020, 1, 1)
    max_start_date = datetime.datetime.today() - datetime.timedelta(days=365)
    start_date = pd.to_datetime(
        st.date_input(
            "Pick a start date for training 👇",
            min_start_date,
            min_value=min_start_date,
            max_value=max_start_date,
        )
    )

    price_data = download_price_data(
        list(sp500.index), start_date=start_date, end_date=datetime.datetime.today()
    )

    spy_prices = download_price_data(
        ["SPY"], start_date=start_date, end_date=datetime.datetime.today()
    )["SPY"]

    diff_assets = set(sp500.index) - set(price_data.columns)
    # assert len(diff_assets) == 0, diff_assets
    if len(diff_assets) != 0:
        print("Data not available for some of the assets!")
        sp500.drop(index=diff_assets, inplace=True)
        sp500["Weight"] /= sp500["Weight"].sum()

    # Calculate daily returns
    returns = price_data.pct_change().dropna()

    # Training data ends at
    min_split_date = start_date + datetime.timedelta(days=365)
    max_split_date = datetime.datetime.today()
    split_date = pd.to_datetime(
        st.date_input(
            "Train/test split date 👇",
            min_split_date,
            min_value=min_split_date,
            max_value=max_split_date,
        )
    )

    # Get covariance matrix
    sigma = returns[returns.index <= split_date].cov() * 252  # Annualized

    ampl = AMPL()
    ampl.eval(TRACKING_ERROR_MODEL)
    ampl.set["ASSETS"] = sp500.index
    ampl.param["w_bench"] = sp500["Weight"]
    ampl.param["sigma"] = sigma
    default_min_weight = ampl.param["min_weight"].value()
    default_max_weight = ampl.param["max_weight"].value()
    default_n_assets = ampl.param["n_assets"].value()
    default_turnover_limit = ampl.param["turnover_limit"].value()
    ampl.param["min_weight"] = st.slider(
        "Minimum weight if invested", 0.0, 0.5, default_min_weight, 0.01
    )
    ampl.param["max_weight"] = st.slider(
        "Maximum weight per asset", 0.0, 0.5, default_max_weight, 0.01
    )
    min_assets = int(1 / ampl.param["max_weight"].value()) + 1
    ampl.param["n_assets"] = st.slider(
        "Target number of holdings", min_assets, 25, default_n_assets, 1
    )
    ampl.param["turnover_limit"] = st.slider(
        "Maximum allowed turnover", 0.0, 1.0, default_turnover_limit, 0.01
    )

    objectives = ["Tracking Error", "Min-Variance", "Both"]
    objective_selected = st.selectbox("Pick the objective 👇", objectives, key="model")
    mp_options = "outlev=1 timelimit=3"
    if objective_selected == "Tracking Error":
        objective = "TrackingErrorSquared"
    elif objective_selected == "Min-Variance":
        objective = "MinVariance"
    else:
        objective = ""
        mp_options += " multiobj=2"

    # Select the solver to use
    solver, _ = solver_selector(mp_only=True)
    # Solve the problem
    output = ampl.solve(
        objective,
        solver=solver,
        mp_options=mp_options,
        return_output=True,
    )
    if ampl.solve_result not in ["solved", "limit"]:
        st.error(f"The model could not be solved:\n```\n{output}\n```")
    else:
        with st.expander("Solver output"):
            st.write(f"```\n{output}\n```")
        w = ampl.var["w"].to_pandas()
        w = w[w["w.val"] >= 0.01]
        w["w.val"] /= w["w.val"].sum()
        plot_pie_chart(pd.Series(ampl.var["w"].to_dict()), "Tracking Error Portfolio")

        normalized_prices = price_data / price_data.iloc[0]
        normalized_spy = spy_prices / spy_prices.iloc[0]

        # Compute portfolio values over time
        benchmark = sp500["Weight"]
        benchmark_returns = (normalized_prices * benchmark).sum(axis=1)
        portfolio = ampl.var["w"].to_pandas()["w.val"]
        portflio_returns = (normalized_prices * portfolio).sum(axis=1)

        # equal weight potfolio
        equal_weight = pd.Series(1 / len(sp500), index=sp500.index)
        equal_weight_returns = (normalized_prices * equal_weight).sum(axis=1)

        portfolio_returns = {
            "SPY": normalized_spy,
            "Benchmark": benchmark_returns,
            "Equal-Weight Portfolio": equal_weight_returns,
            "Our Portfolio": portflio_returns,
        }

        plot_portfolio_comparison(
            split_date,
            portfolio_returns,
        )

    st.markdown(
        """
    #### [[Source Code on GitHub](http://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tracking_error)]
    """
    )
