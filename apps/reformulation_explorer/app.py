import streamlit as st
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

FLAT_CON_TYPES = {
    "_abs": "Abs",
    "_acos": "Acos",
    "_acosh": "Acosh",
    "_alldiff": "AllDiff",
    "_and": "And",
    "_asin": "Asin",
    "_asinh": "Asinh",
    "_atan": "Atan",
    "_atanh": "Atanh",
    "_compl": "Complementarity Linear",
    "_complquad": "Complementarity Quadratic",
    "_condlineq": "Conditional Linear \u2A75",
    "_condlinge": "Conditional Linear \u2A7E",
    "_condlingt": "Conditional Linear >",
    "_condlinle": "Conditional Linear \u2A7D",
    "_condlinlt": "Conditional Linear <",
    "_condquadeq": "Conditional Quadratic \u2A75",
    "_condquadge": "Conditional Quadratic \u2A7E",
    "_condquadgt": "Conditional Quadratic >",
    "_condquadle": "Conditional Quadratic \u2A7D",
    "_condquadlt": "Conditional Quadratic <",
    "_cos": "Cos",
    "_cosh": "Cosh",
    "_count": "Count",
    "_div": "Div",
    "_expa": "ExpA",
    "_exp": "Exp",
    "_expcone": "Exponential Cone",
    "_geomcone": "Geometric Cone",
    "_ifthen": "If-Then",
    "_impl": "Implication \u21D2",
    "_indeq": "Indicator Linear \u2A75",
    "_indge": "Indicator Linear \u2A7E",
    "_indle": "Indicator Linear \u2A7D",
    "_indquadeq": "Indicator Quadratic \u2A75",
    "_indquadge": "Indicator Quadratic \u2A7E",
    "_indquadle": "Indicator Quadratic \u2A7D",
    "_lineq": "Linear \u2A75",
    "_linge": "Linear \u2A7E",
    "_linle": "Linear \u2A7D",
    "_linrange": "Linear Range",
    "_linfunccon": "Linear Function",
    "_loga": "LogA",
    "_log": "Log",
    "_max": "Max",
    "_min": "Min",
    "_not": "Not",
    "_numberofconst": "Numberof/Const",
    "_numberofvar": "Numberof/Var",
    "_or": "Or",
    "_pl": "Piecewise-Linear",
    "_pow": "Pow",
    "_powercone": "Power Cone",
    "_quadeq": "Quadratic \u2A75",
    "_quadge": "Quadratic \u2A7E",
    "_quadle": "Quadratic \u2A7D",
    "_quadrange": "Quadratic Range",
    "_quadcone": "Quadratic Cone",
    "_quadfunccon": "Quadratic Function",
    "_rotatedquadcone": "Rotated Quadratic Cone",
    "_sos1": "SOS1",
    "_sos2": "SOS2",
    "_sin": "Sin",
    "_sinh": "Sinh",
    "_tan": "Tan",
    "_tanh": "Tanh",
    "_uenc": "Unary Encoding",
}


class DiGraph:
    """
    A simple digraph or a wrapper around some graph library
    """

    def __init__(self):
        self._nodes = []
        self._arcs = []

    def add_node(self, data=None):
        self._nodes.append(data)
        return len(self._nodes) - 1

    def get_node(self, idx):
        return self._nodes[idx]

    def to_text(self):
        return str(self._nodes)


class Model:
    """
    An optimization model with conversion graph
    """

    def __init__(self):
        self._graph = DiGraph()  ## Underlyng graph

        self._vars = []  ## Pointers to various parts of the graph
        self._dvars = []
        self._cons_nl_all = []
        self._cons_nl = {  ## NL + SOS
            "All": [],
            "Nonlinear": [],
            "Linear": [],
            "Logical": [],
            "SOS1": [],
            "SOS2": [],
        }
        self._cons_flat = {}
        self._cons_flat_group = {}
        self._objs_nl = []
        self._objs = []

    def update_var(self, idx, data):
        self._update_node_data(self._vars, idx, data)

    def update_def_var(self, idx, data):
        self._update_node_data(self._dvars, idx, data)

    def update_nl_obj(self, idx, data):
        self._update_node_data(self._objs_nl, idx, data)

    def update_flat_obj(self, idx, data):
        self._update_node_data(self._objs, idx, data)

    def update_nl_con(self, type, idx, data):
        if "nonlin" == type:
            self._update_node_data(
                self._cons_nl["Nonlinear"],
                len(self._cons_nl["Nonlinear"]),  ## these just 1x
                data,
            )
        elif "lin" == type:
            self._update_node_data(
                self._cons_nl["Linear"], len(self._cons_nl["Linear"]), data
            )
        elif "logical" == type:
            self._update_node_data(
                self._cons_nl["Logical"], len(self._cons_nl["Logical"]), data
            )
        elif "_sos1" == type:
            self._update_node_data(
                self._cons_nl["SOS1"], len(self._cons_nl["SOS1"]), data
            )
        elif "_sos2" == type:
            self._update_node_data(
                self._cons_nl["SOS2"], len(self._cons_nl["SOS2"]), data
            )
        else:
            raise Exception(f"Unknown NL constraint type: {type}")

    def update_flat_con_group(self, type, data):
        self._cons_flat_group[type] = data

    def update_flat_con(self, type, idx, data):
        if type not in self._cons_flat:
            self._cons_flat[type] = []
        self._update_node_data(self._cons_flat[type], idx, data)
        if (
            0 == data["depth"] and type.startswith("_sos") and "printed" in data
        ):  ## we need the final status
            self.update_nl_con(type, 0, data)

    def _update_node_data(self, specnodecnt, idx, data):
        data1, upd = self._update_item_data(specnodecnt, idx, data)
        if not upd:
            idx = self._graph.add_node(data1)
            data1["node_index"] = idx

    def _update_item_data(self, specnodecnt, idx, data):
        if len(specnodecnt) <= idx:
            specnodecnt.insert(idx, {})
        if specnodecnt[idx] is None:  ## No such item
            specnodecnt[idx] = {}
        ifEmpty = 0 == len(specnodecnt[idx])
        self._update_map(specnodecnt[idx], data)
        return specnodecnt[idx], ifEmpty

    def _update_map(self, data1, data2):
        data1.update(data2)

    # Match keyword to the original model
    def match_orig_model(self, keyw):
        result = {}
        result["NL Variables"] = self._match_records(self._vars, keyw, "is_from_nl")
        result["NL Defined Variables"] = self._match_records(self._dvars, keyw)
        result["NL Objectives"] = self._match_records(self._objs_nl, keyw)
        #    result["NL Constraints"] \
        #      = self._matchRecords(self._cons_nl.get("All"), keyw)
        result["NL Nonlinear Constraints"] = self._match_records(
            self._cons_nl.get("Nonlinear"), keyw
        )
        result["NL Linear Constraints"] = self._match_records(
            self._cons_nl.get("Linear"), keyw
        )
        result["NL Logical Constraints"] = self._match_records(
            self._cons_nl.get("Logical"), keyw
        )
        result["NL SOS1 Constraints"] = self._match_records(
            self._cons_nl.get("SOS1"), keyw
        )
        result["NL SOS2 Constraints"] = self._match_records(
            self._cons_nl.get("SOS2"), keyw
        )
        return result

    # Match keyword to the final model
    def match_final_model(self, keyw):
        result = {}
        result["Variables"] = self._match_records(self._vars, keyw)
        result["Objectives"] = self._match_records(self._objs, keyw)
        for ct, cv in sorted(self._cons_flat.items()):
            result[f"Constraints '{FLAT_CON_TYPES.get(ct, ct)}'"] = self._match_records(
                self._cons_flat[ct], keyw
            )
        return result

    # Add records containing keyword
    # @return array of strings
    def _match_records(self, cnt, keyw, key_need1=None):
        result = ""
        if cnt is None:
            return result
        for i in cnt:
            if "final" not in i or 1 == i["final"]:
                pr = str(i)  ## TODO printed form
                if "printed" in i:
                    pr = i["printed"]
                assert len(pr)
                if ";" != pr[-1]:
                    pr = pr + ";"
                if ("" == keyw or keyw in pr) and (
                    key_need1 == None or (key_need1 in i and 1 == i[key_need1])
                ):
                    result += f"  \n{pr}"  ## Markdown: 2x spaces + EOL
        return result


class ModelView:
    """
    A view of a (sub) model
    """

    def __init__(self):
        self._data = None

        self._vars = {"Variables": []}
        self._cons = {"Constraints": []}
        self._objs = {"Objectives": []}

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return self._data


class Matcher:
    """
    Selects a submodel
    """

    def __init__(self):
        self.data = None


def match_submodel(model: Model, pattern: str, fwd: bool, bwd: bool):
    """
    Match a submodel containg the \a pattern,
    optionally extended by forward/backward
    reformulation graph search
    """
    mv1 = ModelView()
    mv2 = ModelView()
    mv1.set_data(model.match_orig_model(pattern))
    mv2.set_data(model.match_final_model(pattern))
    return mv1, mv2


class ModelReader:
    """
    Model reader
    """

    def __init__(self):
        self._model = Model()

    def read_model(self, uploader):
        for line in uploader:
            # removing the new line characters
            self._process_line(line.rstrip())
        return self._model

    # Process next line
    def _process_line(self, line: str):
        values = json.loads(line)
        self._add_data_chunk(values)

    # Add data chunk as a JSON-like object
    def _add_data_chunk(self, chunk):
        if "VAR_index" in chunk:
            self._model.update_var(chunk["VAR_index"], chunk)
        elif "NL_COMMON_EXPR_index" in chunk:
            self._model.update_def_var(chunk["NL_COMMON_EXPR_index"], chunk)
        elif "NL_OBJECTIVE_index" in chunk:
            self._model.update_nl_obj(chunk["NL_OBJECTIVE_index"], chunk)
        elif "NL_CON_TYPE" in chunk:
            self._model.update_nl_con(chunk["NL_CON_TYPE"], chunk["index"], chunk)
        elif "OBJECTIVE_index" in chunk:
            self._model.update_flat_obj(chunk["OBJECTIVE_index"], chunk)
        elif "CON_GROUP" in chunk:
            self._model.update_flat_con_group(chunk["CON_TYPE"], chunk)
        elif "CON_TYPE" in chunk:
            self._model.update_flat_con(chunk["CON_TYPE"], chunk["index"], chunk)


def read_explorer_model(uploader):
    mr = ModelReader()
    return mr.read_model(uploader)


def main():
    st.header("AMPL MP Reformulation Explorer")
    st.write(
        """
    Documentation: https://mp.ampl.com/modeling-tools.html#reformulation-graph
             
    Usage example:
    - Step 1: Export reformulation graph as follows:
    ```ampl
    param n integer > 0; # N-queens
    var Row {1..n} integer >= 1 <= n;
    s.t. row_attacks: alldiff ({j in 1..n} Row[j]);
    s.t. diag_attacks: alldiff ({j in 1..n} Row[j]+j);
    s.t. rdiag_attacks: alldiff ({j in 1..n} Row[j]-j);

    let n := 5;
    option solver gurobi;
    option gurobi_auxfiles rc; # export row/col names
    option gurobi_options 'writegraph=model.jsonl lim:time=0'; # export graph to model.jsonl 
    solve;
    ```
    - Step 2: Upload the reformulation graph:    
    """
    )

    # To work with local files in st 1.30.1, see
    # https://discuss.streamlit.io/t/axioserror-request-failed-with-status-code-403/38112/13.
    # The corresponding settings should not be used on a server.
    uploader = st.file_uploader(
        "Model graph file (JSONL)",
        help="Reformulation file obtained by option `writegraph`\n"
        + "(https://mp.ampl.com/modeling-tools.html#reformulation-graph)",
    )

    if uploader is None:
        st.write(
            """
        No file selected. Please upload a reformulation file obtained by option 
        `writegraph` (https://mp.ampl.com/modeling-tools.html#reformulation-graph)         
        """
        )
        st.stop()

    search = st.text_input(
        "Search pattern:",
        help="Pattern to filter the models' lines.\nLeave blank to see complete models.",
    )
    fwd = st.checkbox(
        "Add descendants",
        disabled=True,
        help="Include all solver model items derived from matching items in the NL model",
    )
    bwd = st.checkbox(
        "Add ancestors",
        disabled=True,
        help="Include all NL model items reduced to matching items in the solver model",
    )

    left_column, right_column = st.columns(2)

    # Cache the reading function
    @st.cache_data
    def read_model(uploader):
        return read_explorer_model(uploader)

    # Cache the matching function?
    # @st.cache_data  Need cacheable Model.
    def match_selection(m, search, fwd, bwd):
        return match_submodel(m, search, fwd, bwd)

    # Write dictionary of entries
    @st.cache_data
    def write_dict(d):
        whole = ""
        for k, v in d.items():
            if len(v):
                count = v.count("\n")
                whole += f"\n\n##  {k} ({count})\n"
                whole += v
                with st.expander(f"### {k} ({count})"):
                    with st.container(height=200):
                        st.code(v, language="ampl")
        return whole

    filename_upl = ""
    model_nl = ""
    model_flat = ""

    model = read_model(uploader)
    filename_upl = uploader.name
    subm1, subm2 = match_selection(model, search, fwd, bwd)
    bytes1_data = subm1.get_data()
    bytes2_data = subm2.get_data()
    with left_column:
        st.header("NL model", help="NL model lines matching the search pattern")
        model_nl = write_dict(bytes1_data)
    with right_column:
        st.header("Solver model", help="Solver model lines matching the search pattern")
        model_flat = write_dict(bytes2_data)

    st.download_button(
        "Download NL Model",
        f'# NL Model for "{filename_upl}" (search pattern: "{search}")\n{model_nl}',
        f"{filename_upl}_NL.mod",
        help="Download current NL model",
        disabled=("" == model_nl),
    )
    st.download_button(
        "Download Solver Model",
        f'# Solver Model for "{filename_upl}" (search pattern: "{search}")\n{model_flat}',
        f"{filename_upl}_solver.mod",
        help="Download current solver model",
        disabled=("" == model_flat),
    )
