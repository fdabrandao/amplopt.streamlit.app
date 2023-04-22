# AMPL Streamlit Apps

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/fdabrandao/amplopt.streamlit.app/)

[AMPL](https://ampl.com) is the most powerful and intuitive tool for developing and deploying
complex optimization solutions in business & scientific applications.
AMPL connects to most open-source and commercial solvers and allows you to switch easily between them.

AMPL has APIs for several popular programming languages
(e.g., [Python](https://amplpy.readthedocs.io/), [R](https://rampl.readthedocs.io/), etc.)
and it allows you to only model once in AMPL and interact with it using an API for a language 
you are familiar with.

- [AMPL Website](https://ampl.com)
- [AMPL Documentation](https://dev.ampl.com/)
- [AMPL Model Colaboratory](https://colab.ampl.com/)
- [AMPL Forum on Discourse](https://discuss.ampl.com/)

Follow us on [Twitter](https://twitter.com/AMPLopt) and [LinkedIn](https://www.linkedin.com/company/ampl) to get the latest updates from the dev team!

## Build your own app with AMPL & Streamlit

[AMPL and all Solvers are now available as Python Packages](https://dev.ampl.com/ampl/python.html). To use them in [streamlit](https://streamlit.io/) you just need to list the modules in the [requirements.txt](requirements.txt) file as follows:
```
--index-url https://pypi.ampl.com # AMPL's Python Package Index
--extra-index-url https://pypi.org/simple
ampl_module_base # AMPL
ampl_module_highs # HiGHS solver
ampl_module_gurobi # Gurobi solver
amplpy # Python API for AMPL
```

and load them in [streamlit_app.py](streamlit_app.py):
```python
from amplpy import AMPL
ampl = AMPL()
```

- Python API (`amplpy`) documentation: https://amplpy.readthedocs.io/
- Python modules documentation: https://dev.ampl.com/ampl/python/
- AMPL on Streamlit documentation: https://dev.ampl.com/ampl/python/streamlit.html

## How to run it locally

```bash
$ python -m venv venv
$ source venv/bin/activate
$ python -m install -r requirements.txt --upgrade
$ streamlit run streamlit_app.py
```

When you are ready deploy to https://streamlit.io/! This app is running there: https://share.streamlit.io/fdabrandao/amplopt.streamlit.app/