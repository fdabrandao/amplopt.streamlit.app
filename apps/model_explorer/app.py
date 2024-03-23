import streamlit as st
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


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
    An optimization model with conversion
    graph
    """

    def __init__(self):
        self._graph = DiGraph()  ## Underlyng graph

        self._vars = []  ## Pointers to various parts of the graph
        self._dvars = []
        self._cons_NL_all = []
        self._cons_NL = {  ## NL + SOS
            "All": [],
            "Nonlinear": [],
            "Linear": [],
            "Logical": [],
            "SOS1": [],
            "SOS2": [],
        }
        self._cons_Flat = {}
        self._cons_Flat_Group = {}
        self._objs_NL = []
        self._objs = []

    def update_var(self, idx, data):
        self._update_node_data(self._vars, idx, data)

    def update_def_var(self, idx, data):
        self._update_node_data(self._dvars, idx, data)

    def update_nl_obj(self, idx, data):
        self._update_node_data(self._objs_NL, idx, data)

    def update_flat_obj(self, idx, data):
        self._update_node_data(self._objs, idx, data)

    def update_nl_con(self, type, idx, data):
        if "nonlin" == type:
            self._update_node_data(
                self._cons_NL["Nonlinear"],
                len(self._cons_NL["Nonlinear"]),  ## these just 1x
                data,
            )
        elif "lin" == type:
            self._update_node_data(
                self._cons_NL["Linear"], len(self._cons_NL["Linear"]), data
            )
        elif "logical" == type:
            self._update_node_data(
                self._cons_NL["Logical"], len(self._cons_NL["Logical"]), data
            )
        elif "_sos1" == type:
            self._update_node_data(
                self._cons_NL["SOS1"], len(self._cons_NL["SOS1"]), data
            )
        elif "_sos2" == type:
            self._update_node_data(
                self._cons_NL["SOS2"], len(self._cons_NL["SOS2"]), data
            )
        else:
            raise Exception("Unknown NL constraint type: " + type)

    def update_flat_con_group(self, type, data):
        self._cons_Flat_Group[type] = data

    def update_flat_con(self, type, idx, data):
        if type not in self._cons_Flat:
            self._cons_Flat[type] = []
        self._update_node_data(self._cons_Flat[type], idx, data)
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
        result["NL Objectives"] = self._match_records(self._objs_NL, keyw)
        #    result["NL Constraints"] \
        #      = self._matchRecords(self._cons_NL.get("All"), keyw)
        result["NL Nonlinear Constraints"] = self._match_records(
            self._cons_NL.get("Nonlinear"), keyw
        )
        result["NL Linear Constraints"] = self._match_records(
            self._cons_NL.get("Linear"), keyw
        )
        result["NL Logical Constraints"] = self._match_records(
            self._cons_NL.get("Logical"), keyw
        )
        result["NL SOS1 Constraints"] = self._match_records(
            self._cons_NL.get("SOS1"), keyw
        )
        result["NL SOS2 Constraints"] = self._match_records(
            self._cons_NL.get("SOS2"), keyw
        )
        return result

    # Match keyword to the final model
    def match_final_model(self, keyw):
        result = {}
        result["Variables"] = self._match_records(self._vars, keyw)
        result["Objectives"] = self._match_records(self._objs, keyw)
        for ct, cv in sorted(self._cons_Flat.items()):
            result["Constraints '" + ct + "'"] = self._match_records(
                self._cons_Flat[ct], keyw
            )
        return result

    # Add records containing keyword
    # @return array of strings
    def _match_records(self, cnt, keyw, keyNeed1=None):
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
                    keyNeed1 == None or (keyNeed1 in i and 1 == i[keyNeed1])
                ):
                    result = result + "  \n" + pr  ## Markdown: 2x spaces + EOL
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


def match_submodel(m: Model, patt: str, fwd: bool, bwd: bool):
    """
    Match a submodel containg the \a pattern,
    optionally extended by forward/backward
    reformulation graph search
    """
    mv1 = ModelView()
    mv2 = ModelView()
    mv1.set_data(m.match_orig_model(patt))
    mv2.set_data(m.match_final_model(patt))
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
    # To work with local files in st 1.30.1, see
    # https://discuss.streamlit.io/t/axioserror-request-failed-with-status-code-403/38112/13.
    # The corresponding settings should not be used on a server.
    uploader = st.file_uploader(
        "Model file (JSONL)",
        help="Reformulation file obtained by option `writegraph`\n"
        + "(https://mp.ampl.com/modeling-tools.html#reformulation-graph)",
    )

    # You can use a column just like st.sidebar:
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
    def WriteDict(d):
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

    if uploader is not None:
        model = read_model(uploader)
        filename_upl = uploader.name
        subm1, subm2 = match_selection(model, search, fwd, bwd)
        bytes1_data = subm1.get_data()
        bytes2_data = subm2.get_data()
        with left_column:
            st.header("NL model", help="NL model lines matching the search pattern")
            model_nl = WriteDict(bytes1_data)
        with right_column:
            st.header(
                "Solver model", help="Solver model lines matching the search pattern"
            )
            model_flat = WriteDict(bytes2_data)
    else:
        st.header("AMPL MP Reformulation Explorer")
        st.write(
            "Documentation: https://mp.ampl.com/modeling-tools.html#reformulation-graph"
        )
        st.divider()
        st.write("No file selected.")

    st.download_button(
        "Download NL Model",
        f'# NL Model for "{filename_upl}" (search pattern: "{search}")\n{model_nl}',
        filename_upl + "_NL.mod",
        help="Download current NL model",
        disabled=("" == model_nl),
    )
    st.download_button(
        "Download Solver Model",
        f'# Solver Model for "{filename_upl}" (search pattern: "{search}")\n{model_flat}',
        filename_upl + "_solver.mod",
        help="Download current solver model",
        disabled=("" == model_flat),
    )
