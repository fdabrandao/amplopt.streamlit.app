<img src="https://portal.ampl.com/static/img/logo-inline-web-v4.png" align="right" height="60"/>

# Stochastic Facility Location

## Files

- [Streamlit App Source code](app.py)
- [Facility location backend for Nextmv](main.py)
- [AMPL model for facility location model with benders decomposition](floc_bend.mod)

## Streamlit

Install requirements:

```bash
pip3 install -r requirements.txt
```

Run the command below to start the streamlit app:

```bash
streamlit run app.py
```

## Nextmv

### Usage example (local)

Install requirements:

```bash
pip3 install -r requirements.txt
```

Run the command below to check that everything works as expected:

```bash
python3 main.py -input input.json -output output.json -duration 30 -provider highs
```

A file `output.json` should have been created with the solution.

### Usage example (remote)

Push the app to Nextmv:

```bash
nextmv app push -a facility-location
```

Make a run:

```bash
nextmv app run -a facility-location -i input.json -w > output.json
```

Create a new instance with a new version of the app:

```bash
VERSION_ID=$(git rev-parse --short HEAD)
nextmv app version create \
    -a facility-location \
    -n $VERSION_ID \
    -v $VERSION_ID
nextmv app instance create \
    -a facility-location \
    -v $VERSION_ID \
    -i candidate-1 \
    -n "Test candidate 1"
```
