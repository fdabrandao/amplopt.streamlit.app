from numbers import Real
from collections.abc import Iterable
import pandas as pd
import numpy as np
import json


def py_cast(value):
    if isinstance(value, (Real, str, list, dict)):
        return value
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        return list(map(tuple, value.reset_index().itertuples(index=False)))
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, Iterable):
        return list(value)
    return value


def table_to_dict(rows):
    if len(rows) == 0:
        return {}
    assert len(rows[0]) >= 2
    return {tuple(row[:-1]) if len(row) > 2 else row[0]: row[-1] for row in rows}


def index_to_key(index):
    if isinstance(index, Iterable) and len(index) == 1:
        return index[0]
    return index


def dict_to_table(d):
    return [(k if isinstance(k, tuple) else (k,)) + (v,) for k, v in d.items()]


def set_py_to_json(values):
    if isinstance(values, dict):  # indexed set
        return [[[k], v] for k, v in values.items()]
    else:
        return values


def set_json_to_py(values):
    if (
        isinstance(values, list) and len(values) > 0 and isinstance(values[0], list)
    ):  # indexed set
        return {index_to_key(k): v for k, v in values}
    else:
        return values


def param_py_to_json(values):
    if isinstance(values, dict):
        return dict_to_table(values)
    else:
        return values


def param_json_to_py(values):
    if isinstance(values, list):
        return table_to_dict(values)
    else:
        return values


class DataSerializer:
    def __init__(self, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = {"sets": {}, "params": {}}

    def _set(self):
        class Sets(object):
            def __init__(self, sets):
                self.sets = sets

            def __getitem__(self, name):
                return self.sets[name]

            def __setitem__(self, name, values):
                if isinstance(values, dict):
                    self.sets[name] = {
                        index_to_key(k): py_cast(v) for k, v in values.items()
                    }
                else:
                    self.sets[name] = py_cast(values)

            def __iter__(self):
                return self.sets

        return Sets(self.data["sets"])

    def _param(self):
        class Parameters(object):
            def __init__(self, params):
                self.params = params

            def __getitem__(self, name):
                return self.params[name]

            def __setitem__(self, name, values):
                values = py_cast(values)
                if isinstance(values, list):
                    values = table_to_dict(values)

                if isinstance(values, (Real, str)):
                    self.params[name] = values
                elif isinstance(values, dict):
                    self.params[name] = values
                else:
                    raise ValueError("Unexpected data type")

            def __iter__(self):
                return self.params

        return Parameters(self.data["params"])

    set = property(_set)
    param = property(_param)

    @classmethod
    def from_json(cls, json_data):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        sets, params = json_data.get("sets", {}), json_data.get("params", {})
        return cls(
            {
                "sets": {name: set_json_to_py(values) for name, values in sets.items()},
                "params": {
                    name: param_json_to_py(values) for name, values in params.items()
                },
            }
        )

    def to_json_obj(self):
        return {
            "sets": {
                name: set_py_to_json(values)
                for name, values in self.data["sets"].items()
            },
            "params": {
                name: param_py_to_json(values)
                for name, values in self.data["params"].items()
            },
        }

    def to_json(self):
        return json.dumps(self.to_json_obj())

    def to_dat(self):
        dat = "data;"

        def set_members_to_dat(members):
            return " ".join(map(str, members))

        def param_to_dat(values):
            if not isinstance(values, dict):
                return str(values)
            return " ".join(
                map(
                    lambda x: (
                        f"{x[0]} {x[1]}"
                        if not isinstance(x[0], tuple)
                        else f"{' '.join(x[0])} {x[1]}"
                    ),
                    values.items(),
                )
            )

        for name, values in self.data["sets"].items():
            if isinstance(values, list):
                dat += f"set {name} := {set_members_to_dat(values)};\n"
            else:
                for index, members in values.items():
                    dat += f"set {name}[{index}] := {set_members_to_dat(members)};\n"
        for name, values in self.data["params"].items():
            dat += f"param {name} := {param_to_dat(values)};\n"
        return dat


class TableSerializer:
    def __init__(self, values=None):
        if not isinstance(values, dict):
            values = py_cast(values)
            if isinstance(values, list):
                values = table_to_dict(values)
        self.values = values

    def to_dict(self):
        return self.values

    @classmethod
    def from_json(cls, json_data):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        return cls(param_json_to_py(json_data))

    def to_json_obj(self):
        return param_py_to_json(self.values)

    def to_json(self):
        return json.dumps(self.to_json_obj())
