<img src="https://portal.ampl.com/static/img/logo-inline-web-v4.png" align="right" height="60"/>

# Batch Process Optimization

## Files

- [Streamlit App Source code](app.py)
- [Batch Process Optimization backend for Nextmv](main.py)
- [AMPL model for Batch Process Optimization](floc_bend.mod)

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
nextmv app push -a batch-process
```

Make a run:

```bash
nextmv app run -a batch-process -i input.json -w > output.json
```

Create a new instance with a new version of the app:

```bash
VERSION_ID=$(git rev-parse --short HEAD)
nextmv app version create \
    -a batch-process \
    -n $VERSION_ID \
    -v $VERSION_ID
nextmv app instance create \
    -a batch-process \
    -v $VERSION_ID \
    -i candidate-4 \
    -n "Test candidate 4"
```
