import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from amplpy import AMPL
import matplotlib.pyplot as plt
import seaborn as sns
from ..common import solver_selector, MP_SOLVERS_LINKS

TRACKING_ERROR_MODEL = r"""
# Sets
set ASSETS;  # Set of assets

# Parameters
param sigma{ASSETS, ASSETS};  # Covariance matrix of asset returns
param w_prev{ASSETS} default 0;  # Portfolio weights of the previous period
param w_bench{ASSETS};  # Weights of assets in the benchmark portfolio
param min_weight default 0.01;  # Minimum weight accepted
param max_weight default 0.20;  # Maximum weight accepted

# Decision Variables
var w{ASSETS} >= 0, <= max_weight;  # Portfolio weights (long-only)

# Objective: Minimize Tracking Error
minimize TrackingErrorSquared:
    sum{i in ASSETS, j in ASSETS} (w[i] - w_bench[i]) * sigma[i,j] * (w[j] - w_bench[j]);

s.t. FullyInvested:
    sum{i in ASSETS} w[i] = 1;  # Fully invested portfolio

s.t. PositionCount:
    count {i in ASSETS} (w[i] != 0) = 10;  # Exactly 10 positions

s.t. MinWeightOrZero{i in ASSETS}:
    w[i] = 0 or w[i] >= min_weight;  # Either zero or above minimum

s.t. Turnover{if sum{i in ASSETS} w_prev[i] > 0}:
    sum{i in ASSETS} abs(w[i] - w_prev[i]) <= 0.05;  # Turnover constraint
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


def plot_portfolio_comparison(prices, benckmark, porfolio):
    """Plot comparison of portfolio returns."""
    normalized_prices = prices / prices.iloc[0]

    # Compute portfolio values over time
    pf1_returns = (normalized_prices * benckmark).sum(axis=1)
    pf2_returns = (normalized_prices * porfolio).sum(axis=1)

    # Combine into a single DataFrame for plotting
    combined = pd.DataFrame(
        {"Benchmark": pf1_returns, "Tracking Error Portfolio": pf2_returns}
    )

    # Seaborn style
    sns.set(style="whitegrid")

    # Plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=combined, ax=ax)
    ax.set_title("Portfolio Performance Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")

    # Show in Streamlit
    st.pyplot(fig)


def main():
    st.title("📈 Tracking Error Minimization")

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

    st.write("## Top 100 from current S&P 500 constituents")
    st.dataframe(sp500)

    price_data = download_price_data(
        list(sp500.index), start_date="2024-01-01", end_date="2025-01-01"
    )

    diff_assets = set(sp500.index) - set(price_data.columns)
    # assert len(diff_assets) == 0, diff_assets
    if len(diff_assets) != 0:
        print("Data not available for some of the assets!")
        sp500.drop(index=diff_assets, inplace=True)
        sp500["Weight"] /= sp500["Weight"].sum()

    st.write("## Model")
    st.code(TRACKING_ERROR_MODEL)

    # Calculate daily returns
    returns = price_data.pct_change().dropna()

    # Get covariance matrix
    sigma = returns.cov() * 252  # Annualized

    ampl = AMPL()
    ampl.eval(TRACKING_ERROR_MODEL)
    ampl.set["ASSETS"] = sp500.index
    ampl.param["w_bench"] = sp500["Weight"]
    ampl.param["sigma"] = sigma

    # Select the solver to use
    solver, _ = solver_selector(mp_only=True)
    # Solve the problem
    output = ampl.solve(
        solver=solver, mp_options="outlev=1 timelimit=3", return_output=True
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

        plot_portfolio_comparison(
            price_data, sp500["Weight"], ampl.var["w"].to_pandas()["w.val"]
        )

    st.markdown(
        """
    #### [[Source Code on GitHub](http://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/tracking_error)]
    """
    )
