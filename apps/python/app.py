import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    st.markdown(
        '''
    # Use AMPL from 🐍 easily on:

    - [Local machines](#on-local-machines)
    - [Google Colab](#on-google-colab)
    - [Streamlit Cloud](#on-streamlit-cloud)
    - [Docker Containers](#on-docker-containers)
    - [Continuous Integration (CI/CD) systems](#on-continuous-integration-ci-cd-systems)

    ## On local machines

    [AMPL and all solvers are now available as python packages](https://dev.ampl.com/ampl/python/modules.html#amplpy-modules) for **Windows, Linux (X86_64, aarch64, ppc64le), and macOS**. For instance, to install AMPL with HiGHS, CBC and Gurobi, you just need the following:

    ```bash
    # Install Python API for AMPL
    $ python -m pip install amplpy --upgrade

    # Install solver modules (e.g., HiGHS, CBC, Gurobi)
    $ python -m amplpy.modules install highs cbc gurobi

    # Activate your license (e.g., free https://ampl.com/ce license)
    $ python -m amplpy.modules activate <license-uuid>

    # Import in Python
    $ python
    >>> from amplpy import AMPL
    >>> ampl = AMPL() # instantiate AMPL object

    ```

    ---

    You can use a free [Community Edition license](https://ampl.com/ce), which allows **free and perpetual use of AMPL with Open-Source solvers**.
    
    ---
    
    ```python
    # Minimal example:
    from amplpy import AMPL
    import pandas as pd
    ampl = AMPL()
    ampl.eval(r"""
        set A ordered;
        param S{A, A};
        param lb default 0;
        param ub default 1;
        var w{A} >= lb <= ub;
        minimize portfolio_variance:
            sum {i in A, j in A} w[i] * S[i, j] * w[j];
        s.t. portfolio_weights:
            sum {i in A} w[i] = 1;
    """)
    tickers, cov_matrix = # ... pre-process data in Python
    ampl.set["A"] = tickers
    ampl.param["S"] = pd.DataFrame(
        cov_matrix, index=tickers, columns=tickers
    )
    ampl.option["solver"] = "gurobi"
    ampl.option["gurobi_options"] = "outlev=1"
    ampl.solve()
    assert ampl.get_value("solve_result") == "solved"
    sigma = ampl.get_value("sqrt(sum {i in A, j in A} w[i] * S[i, j] * w[j])")
    print(f"Volatility: {sigma*100:.1f}%")
    # ... post-process solution in Python

    ```

    [[Python API](http://amplpy.readthedocs.io)] [[GitHub](https://github.com/ampl/amplpy)]

    ## On Google Colab

    You can also install AMPL on [Google Colab](https://dev.ampl.com/ampl/python/colab.html) ([where it is free by default](https://colab.ampl.com/getting-started.html#ampl-is-free-on-colab)) as follows:

    ```python
    # Install dependencies
    !pip install -q amplpy
    # Google Colab & Kaggle integration
    from amplpy import AMPL, tools
    ampl = tools.ampl_notebook(
        modules=["coin", "highs", "gokestrel"], # modules to install
        license_uuid="default", # license to use
        g=globals()) # instantiate AMPL object and register magics

    ```

    ---

    On Google Colab there is a default [AMPL Community Edition license](https://ampl.com/ce/) that gives you **unlimited access to AMPL with open-source solvers** (e.g., HiGHS, CBC, Couenne, Ipopt, Bonmin) or with commercial solvers from the [NEOS Server](http://www.neos-server.org/) as described in [Kestrel documentation](https://dev.ampl.com/solvers/kestrel.html).

    In the list `modules` you need to include `"gokestrel"` to use the [kestrel](https://dev.ampl.com/solvers/kestrel.html) driver; `"highs"` for the [HiGHS](https://highs.dev/) solver; `"coin"` for the [COIN-OR](https://www.coin-or.org/) solvers. To use other commercial solvers, your license needs to include the commercial solver (e.g., an AMPL CE commercial solver trial).

    ---
    
    [[Model Colaboratory](http://colab.ampl.com)]

    ## On Streamlit Cloud

    AMPL can be used on [Streamlit](https://streamlit.io/) to produce interactive prescriptive analytics applications easily.

    Check it out on Streamlit Cloud: [![RunOnStreamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://ampl.com/streamlit)

    -   [💡 Modeling tips on Streamlit](http://ampl.com/streamlit/Modeling_Tips)
        
    -   [👑 M-Queens Problem](http://ampl.com/streamlit/N-Queens)
        
    -   [📈 Risk Return (Prescriptive Analytics example)](http://ampl.com/streamlit/Risk_Return)

    Since AMPL and all Solvers are now available as [Python Packages](http://dev.ampl.com/ampl/python/modules.html). To use them in [streamlit](https://streamlit.io) you just need to list the modules in the [requirements.txt](https://github.com/fdabrandao/amplopt.streamlit.app/blob/master/requirements.txt) file as follows:

    ```
    --index-url https://pypi.ampl.com # AMPL's Python Package Index
    --extra-index-url https://pypi.org/simple
    ampl_module_base # AMPL
    ampl_module_highs # HiGHS solver
    ampl_module_gurobi # Gurobi solver
    amplpy # Python API for AMPL
    ```

    and load them in [streamlit_app.py](https://github.com/fdabrandao/amplopt.streamlit.app/blob/master/streamlit_app.py):
    ```python
    from amplpy import AMPL
    ampl = AMPL()
    ```

    [[AMPL on Streamlit](http://ampl.com/streamlit)] [[Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app)]

    ## On Docker Containers

    AMPL can be easily used on [Docker containers](https://www.docker.com/).
    On Python containers the setup is the easiest
    since AMPL and all solvers are now available as [Python Packages](https://dev.ampl.com/ampl/python/modules.html):

    ```bash
    # Use any image as base image with python installed
    FROM python:3.9-slim-bullseye

    # Install amplpy and all necessary amplpy.modules:
    RUN python -m pip install amplpy --no-cache-dir # Install amplpy
    RUN python -m amplpy.modules install highs gurobi --no-cache-dir # Install modules
    ```
    We do not provide a base docker image as we give the user total freedom about which base image to use.
    In this example, we use the image [`python:3.9-slim-bullseye`](https://hub.docker.com/_/python) as base image.

    You can build and run the container locally as follows:
    ```bash
    $ docker build . --tag ampl-container
    $ docker run --rm -it ampl-container bash
    root@c240a014dd67:/# python
    Python 3.9.16 (main, Jan 23 2023, 23:42:27)
    [GCC 10.2.1 20210110] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from amplpy import AMPL, modules
    >>> modules.activate("<license-uuid>")
    >>> ampl = AMPL()
    >>>
    ```

    This also works with non-root containers:
    ```bash
    # Use any image as base image
    FROM python:3.9-slim-bullseye

    # Install amplpy and all necessary amplpy.modules:
    RUN python -m pip install amplpy --no-cache-dir # Install amplpy
    RUN python -m amplpy.modules install highs gurobi --no-cache-dir # Install modules

    # Add non-root user (optional)
    ARG USERNAME=guest
    ARG USER_UID=1000
    ARG USER_GID=$USER_UID
    RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    # Change to non-root privilege
    USER ${USERNAME}
    ```

    ## On Continuous Integration (CI/CD) systems

    AMPL can be easily used in [Continuous Integration/Continuous Delivery (CI/CD)](https://en.wikipedia.org/wiki/CI/CD) platforms in order to test your optimization applications before deploying to production.

    Since AMPL and all solvers are now available as Python Packages. It is only necessary to have a step which installs [amplpy](https://amplpy.readthedocs.io) and the modules needed by your application:

    ```bash
    python -m pip install amplpy
    python -m amplpy.modules install <solver1> <solver2> # install the solvers you need
    python -m amplpy.activate <license-uuid> # activate your license
    ```

    Below, we have examples for two popular CI/CD platforms: [GitHub Actions](https://github.com/features/actions) and [Azure Pipelines](https://azure.microsoft.com/en-us/products/devops/pipelines). The setup is very similar for others.

    ### GitHub Actions

    [Example configuration for GitHub Actions](https://github.com/ampl/amplpyfinance/blob/master/.github/workflows/test.yaml) used in the example project [amplpyfinance](https://github.com/ampl/amplpyfinance).

    File `".github/workflows/test.yaml"`:
    ```yaml
    name: Test
    run-name: ${{ github.actor }} is building "${{ github.ref_name }}"
    on: [push]

    jobs:
    Test:
        runs-on: ubuntu-latest
        strategy:
        matrix:
            python-version: ["3.10"]

        steps:
        - uses: actions/checkout@v3
        - name: Set up Python ${{ matrix.python-version }}
            uses: actions/setup-python@v4
            with:
            python-version: ${{ matrix.python-version }}
        - name: Install dependencies
            run: |
            set -ex
            python -m pip install -r requirements.txt
            python -m pip install amplpy
            python -m amplpy.modules install <solver1> <solver2>
            python -m amplpy.activate <license-uuid>
        - name: Install package
            run: |
            python -m pip install .
        - name: Test package
            run: |
            python -m <package-name>.tests
    ```

    ### Azure Pipelines

    [Example configuration for Azure Pipelines](https://github.com/ampl/amplpyfinance/blob/master/azure-pipelines.yml) used in the example project [amplpyfinance](https://github.com/ampl/amplpyfinance).

    File `"azure-pipelines.yml"`:
    ```yaml
    jobs:
    - job: Test
    pool: {vmImage: 'ubuntu-latest'}
    strategy:
        matrix:
        Python 3.10:
            PYTHON_VERSION: 3.10
    steps:
        - task: UsePythonVersion@0
        inputs:
            versionSpec: $(PYTHON_VERSION)
        - bash: python --version
        displayName: Check python version
        - bash: |
            set -ex
            python -m pip install -r requirements.txt
            python -m pip install amplpy
            python -m amplpy.modules install <solver1> <solver2>
            python -m amplpy.activate <license-uuid>
        displayName: Install dependencies
        - bash: |
            python -m pip install .
        displayName: Install package
        - bash: |
            python -m <package-name>.tests
        displayName: Test package
    ```


    ---

    **Contact us at <support@ampl.com> for free deployment support.** You can also ask any questions you may have on our [Support Forum](https://discuss.ampl.com).

    ---
    '''
    )
