"""
Template for working with AMPL.
"""

import argparse
import json
import os
import sys
import time
from platform import uname
from typing import Any
import pandas as pd
import io

from serializer import DataSerializer, TableSerializer


from amplpy import AMPL, ErrorHandler, OutputHandler, modules

# Duration parameter for the solver.
SUPPORTED_PROVIDER_DURATIONS = {
    "cbc": "timelimit",
    "copt": "timelimit",
    "gcg": "timelimit",
    "gurobi": "timelimit",
    "highs": "timelimit",
    "lgo": "timelim",
    "scip": "timelimit",
    "xpress": "timelimit",
}


# Status of the solver after optimizing.
STATUS = [
    {"lb": 0, "ub": 99, "status": "optimal"},
    {"lb": 100, "ub": 199, "status": "solved?"},
    {"lb": 200, "ub": 299, "status": "infeasible"},
    {"lb": 300, "ub": 399, "status": "unbounded"},
    {"lb": 400, "ub": 499, "status": "limit"},
    {"lb": 500, "ub": 599, "status": "failure"},
]


class CollectOutput(OutputHandler):
    def __init__(self):
        self.buffer = ""

    def output(self, kind, msg):
        self.buffer += msg


output_handler = CollectOutput()


class CollectWarnings(ErrorHandler):
    def __init__(self):
        self.buffer = ""

    def warning(self, exception):
        self.buffer += str(exception)


error_handler = CollectWarnings()


def main() -> None:
    """Entry point for the template."""

    parser = argparse.ArgumentParser(description="Solve problems with AMPL.")
    parser.add_argument(
        "-input",
        default="",
        help="Path to input file. Default is stdin.",
    )
    parser.add_argument(
        "-output",
        default="",
        help="Path to output file. Default is stdout.",
    )
    parser.add_argument(
        "-duration",
        default=30,
        help="Max runtime duration (in seconds). Default is 30.",
        type=int,
    )
    parser.add_argument(
        "-provider",
        default="highs",
        help="Solver provider. Default is highs.",
    )
    args = parser.parse_args()

    # Read input "data", solve the problem and write the solution.
    input_data = read_input(args.input)
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    log("Solving stochastic facility location problem:")
    log(f"  - facilities: {input_data.get('FACILITIES', [])}")
    log(f"  - customers: {input_data.get('CUSTOMERS', 0)}")
    log(f"  - max duration: {args.duration} seconds")
    solution = solve(input_data, args.duration, args.provider)
    write_output(args.output, solution)


def solve(input_data: dict[str, Any], duration: int, provider: str) -> dict[str, Any]:
    """Solves the given problem and returns the solution."""

    start_time = time.time()

    # Activate license.
    license_used = activate_license()

    # Defines the model.
    ampl = AMPL()
    ampl.set_output_handler(output_handler)
    ampl.set_error_handler(error_handler)
    ampl = AMPL()
    ampl.read("batch_process.mod")
    ds = DataSerializer.from_json(input_data)
    ampl.eval(ds.to_dat())
    ampl.option["highs_options"] = "outlev=1"
    ampl.option["gurobi_options"] = "outlev=1"
    ampl.option["cplex_options"] = "outlev=1"

    # Sets the solver and options.
    ampl.option["solver"] = provider
    if provider in SUPPORTED_PROVIDER_DURATIONS.keys():
        opt_name = f"{provider}_options"
        if not ampl.option[opt_name]:
            ampl.option[opt_name] = ""
        ampl.option[opt_name] += f" {SUPPORTED_PROVIDER_DURATIONS[provider]}={duration}"
    solve_output = ampl.solve(return_output=True)

    log(f"AMPL output after solving: {output_handler.buffer}")
    log(f"AMPL errors after solving: {error_handler.buffer}")

    # Creates the statistics.
    statistics = {
        "result": {
            "custom": {
                "provider": provider,
                "status": ampl.solve_result,
                "variables": ampl.get_value("_nvars"),
                "constrains": ampl.get_value("_ncons"),
            },
            "duration": ampl.get_value("_total_solve_time"),
            "value": ampl.get_value("Total_Profit"),
        },
        "run": {
            "duration": time.time() - start_time,
            "custom": {
                "license_used": license_used,
            },
        },
        "schema": "v1",
    }

    return {
        "solutions": [
            {
                "total_value": ampl.get_value("TotalValue"),
                "total_cost": ampl.get_value("TotalCost"),
                "total_profit": ampl.get_value("Total_Profit"),
                "W": TableSerializer(ampl.var["W"].to_pandas()).to_json_obj(),
                "B": TableSerializer(ampl.var["B"].to_pandas()).to_json_obj(),
                "S": TableSerializer(ampl.var["S"].to_pandas()).to_json_obj(),
                "Q": TableSerializer(ampl.var["Q"].to_pandas()).to_json_obj(),
                "solve_output": solve_output,
                "solve_result": ampl.solve_result,
                "solve_time": ampl.get_value("_total_solve_time"),
            }
        ],
        "statistics": statistics,
    }


def activate_license() -> str:
    """
    Activates de AMPL license based on the use case for the app. If there is a
    license configured in the file, and it is different from the template
    message, it activates the license. Otherwise, use a special module if
    running on Nextmv Cloud. No further action required for testing locally.

    Returns:
        str: The license that was activated: "license", "nextmv" or
        "not_activated".
    """

    # Check if the ampl_license_uuid file exists. NOTE: When running in Nextmv
    # Cloud with a valid license, make sure to run on a premium execution
    # class. Contact support for more information.
    if os.path.isfile("ampl_license_uuid"):
        with open("ampl_license_uuid") as file:
            license = file.read().strip()

        # If the license is not the template message, activate it.
        if license != "secret-key-123":
            modules.activate(license)
            return "license"

    # A valid AMPL license has not been configured. When running on Nextmv
    # Cloud, use a special module.
    system_info = uname()
    if system_info.system == "Linux" and "aarch64" in system_info.machine:
        modules.activate("nextmv")
        return "nextmv"

    return "demo"


def log(message: str) -> None:
    """Logs a message. We need to use stderr since stdout is used for the
    solution."""

    print(message, file=sys.stderr)


def read_input(input_path) -> dict[str, Any]:
    """Reads the input from stdin or a given input file."""

    input_file = {}
    if input_path:
        with open(input_path, encoding="utf-8") as file:
            input_file = json.load(file)
    else:
        input_file = json.load(sys.stdin)

    return input_file


def write_output(output_path, output) -> None:
    """Writes the output to stdout or a given output file."""

    content = json.dumps(output, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
