import streamlit as st
from amplpy import AMPL
import io
import contextlib
import math
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# pyplot_show = lambda plt: plt.show()
# pyplot_show = lambda plt: st.pyplot(plt)
# display = lambda x: st.write(x)


def split_preferences(pref):
    desired_slots = [s for s in pref if pref[s] > 0]
    desired_slots.sort(key=lambda s: pref[s], reverse=True)
    undesired_slots = [s for s in pref if pref[s] == 0]
    return desired_slots, undesired_slots


class Instance:
    def __init__(self, init_dict=None):
        ## Data arrays / matrices for all trainees
        self.num_trainees = 0
        self.num_sessions = 0
        self.trainees = []
        self.sessions = []
        self.trainee_position = {}
        self.trainee_seniority = {}
        self.trainee_language = {}
        self.trainee_expiration = {}
        self.trainee_preferences = {}
        ## Capacity constraints
        self.positions = []
        self.languages = []
        self.position_capacity = {}
        self.meta_positions = []
        self.position_groups = {}
        self.group_capacity = {}
        if init_dict is not None:
            for key, value in init_dict.items():
                self.__dict__[key] = value

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_data):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        return cls(json_data)

    def instance_editor(self):
        df = pd.DataFrame(
            {
                "Position": self.trainee_position,
                "Seniority": self.trainee_seniority,
                "Language": self.trainee_language,
                "Expiration": self.trainee_expiration,
            }
        ).reset_index()
        df.rename(columns={"index": "Trainee"}, inplace=True)
        df = st.data_editor(
            df,
            disabled=["Trainee"],
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Position": st.column_config.SelectboxColumn(
                    "💼 Position",
                    help="The position of the trainee",
                    options=self.positions,
                    required=True,
                ),
                "Seniority": st.column_config.NumberColumn(
                    "👴 Seniority",
                    help="The seniority of the trainee",
                    min_value=0,
                    max_value=len(self.trainees) - 1,
                    step=1,
                    required=1,
                ),
                "Language": st.column_config.SelectboxColumn(
                    "🗣️ Language",
                    help="The language of the trainee",
                    options=self.languages,
                    required=True,
                ),
                "Expiration": st.column_config.SelectboxColumn(
                    "⏳ Expiration",
                    help="Flight capability expiration",
                    options=[0, 1, 2],
                    required=True,
                ),
            },
        ).set_index(["Trainee"])
        self.trainee_position = df["Position"].to_dict()
        self.trainee_seniority = df["Seniority"].to_dict()
        self.trainee_language = (
            df["Language"].apply(lambda l: self.languages.index(l)).to_dict()
        )
        self.trainee_expiration = df["Expiration"].to_dict()

        with st.expander("📅 Training Slot Preferences"):
            for t in self.trainee_preferences:
                desired_slots, undesired_slots = split_preferences(
                    self.trainee_preferences[t]
                )
                left, right = st.columns(2)
                pref = {}
                with left:
                    desired_slots = st.multiselect(
                        f"Desired slots for trainee {t} 👇",
                        self.sessions,
                        default=desired_slots,
                    )
                    for i, s in enumerate(desired_slots):
                        pref[s] = len(desired_slots) - i
                with right:
                    sessions_left = self.sessions[:]
                    for s in desired_slots:
                        sessions_left.remove(s)
                    undesired_slots = st.multiselect(
                        f"Valid but undesired slots for trainee {t} 👇",
                        sessions_left,
                        default=undesired_slots,
                    )
                    for s in undesired_slots:
                        pref[s] = 0
                self.trainee_preferences[t] = pref


class InstanceGenerator:
    def __init__(self, num_trainees: int, num_sessions: int, rng: np.random.Generator):
        self.rng = rng
        self.num_trainees = num_trainees
        self.num_sessions = num_sessions  # Set lower to have non-assignments
        ## Ranges and probabilities
        self.position_prob = [0.25, 0.25, 0.25, 0.25]
        self.language_prob = [0.5, 0.25, 0.25]
        self.expiration_prob = [0.5, 0.25, 0.25]
        self.num_sessions_range = [1, math.floor(self.num_sessions / 2)]
        self.pref_0 = 0.5
        ## These are constant for all instances
        self.languages = ["Both", "Language 1", "Language 2"]
        CP, FO, PU, FA = (
            "CP: Captain",
            "FO: First Officer",
            "PU: Purser",
            "FA: Flight Attendant",
        )
        CK, CB = "CK: Cockpit", "CB: Cabin"
        self.positions = [CP, FO, PU, FA]
        self.position_capacity = {CP: 4, FO: 4, PU: 4, FA: 4}
        self.meta_positions = ["All", CK, CB]
        self.group_capacity = {"All": 10, CK: 6, CB: 6}
        self.position_groups = {
            CP: ["All", CK],
            FO: ["All", CK],
            PU: ["All", CB],
            FA: ["All", CB],
        }

    def generator_editor(self):
        left, right = st.columns(2)

        def probability_editor(label, indices, values):
            df = st.data_editor(
                pd.DataFrame(
                    {
                        label: indices,
                        "Probability": values,
                    }
                ).set_index([label]),
                column_config={
                    "Probability": st.column_config.NumberColumn(required=True)
                },
            )
            df["Probability"] /= df["Probability"].sum()
            return df["Probability"].tolist()

        with left:
            self.position_prob = probability_editor(
                "Position", self.positions, self.position_prob
            )
        with right:
            self.language_prob = probability_editor(
                "Language", self.languages, self.language_prob
            )
        with left:
            self.expiration_prob = probability_editor(
                "Expiration", range(len(self.expiration_prob)), self.expiration_prob
            )
        with right:
            self.num_sessions_range[0] = st.slider(
                "Minimum number of valid sessions 👇",
                min_value=1,
                max_value=self.num_sessions,
                value=self.num_sessions_range[0],
                step=1,
            )
            self.num_sessions_range[1] = st.slider(
                "Maximum number of valid sessions 👇",
                min_value=self.num_sessions_range[0],
                max_value=self.num_sessions,
                value=self.num_sessions_range[1],
                step=1,
            )
            self.pref_0 = st.slider(
                "Probability of undesired slots 👇",
                min_value=0.0,
                max_value=1.0,
                value=self.pref_0,
            )

    def generate_instance(self):
        rng = self.rng
        inst = Instance()
        inst.num_trainees = self.num_trainees
        inst.num_sessions = self.num_sessions
        inst.trainees = [f"T{i+1}" for i in range(self.num_trainees)]
        inst.sessions = [f"S{i+1}" for i in range(self.num_sessions)]

        seniorities = np.arange(self.num_trainees)  # Seniority 0..N-1
        rng.shuffle(seniorities)
        inst.trainee_seniority = dict(
            zip(
                inst.trainees,
                map(int, seniorities),
            )
        )
        for _, t in enumerate(inst.trainees):
            inst.trainee_position[t] = rng.choice(
                self.positions,
                p=self.position_prob,
            )
            inst.trainee_language[t] = rng.choice(self.languages, p=self.language_prob)

            inst.trainee_expiration[t] = int(
                rng.choice(len(self.expiration_prob), p=self.expiration_prob)
            )
            number_feasible_sessions = rng.integers(
                self.num_sessions_range[0], self.num_sessions_range[1] + 1
            )
            valid_sessions = list(
                rng.choice(
                    inst.sessions,
                    size=number_feasible_sessions,
                    replace=False,
                )
            )
            num_pref_zero = rng.binomial(number_feasible_sessions, self.pref_0)
            preference_score = np.concatenate(
                (
                    np.zeros(num_pref_zero),
                    np.arange(1, number_feasible_sessions - num_pref_zero + 1),
                )
            )
            assert number_feasible_sessions == len(preference_score)
            rng.shuffle(preference_score)
            inst.trainee_preferences[t] = dict(zip(valid_sessions, preference_score))
        # Constant
        inst.languages = self.languages
        inst.positions = self.positions
        inst.position_capacity = self.position_capacity
        inst.meta_positions = self.meta_positions
        inst.position_groups = self.position_groups
        inst.group_capacity = self.group_capacity
        return inst


def make_ampl_instance(models: list, inst: Instance):
    ampl = AMPL()
    ampl.cd(os.path.dirname(__file__))
    for mod in models:
        ampl.read(mod)
    # Trainees
    ampl.set["Trainees"] = inst.trainees

    # Training sessions
    ampl.set["Sessions"] = inst.sessions

    # Positons (e.g., captain (CP), first officer (FO), purser (PU), or flight attendant (FA))
    ampl.set["Positions"] = inst.positions

    # Meta-positions (e.g, All, Cockpit (CK), Cabin (CB))
    ampl.set["MetaPositions"] = inst.meta_positions

    # Groups corresponding to each position
    ampl.set["PositionGroups"] = inst.position_groups

    # Trainee's position
    ampl.param["TraineePosition"] = inst.trainee_position

    # TraineeSeniority (smaller value <=> higher seniority)
    ampl.param["TraineeSeniority"] = inst.trainee_seniority

    # TraineeLanguage (0 - both, 1 or 2 - one only)
    ampl.param["TraineeLanguage"] = inst.trainee_language

    # TraineeExpiration: 0 - this month, 1 - next month, 2 - in 2 months
    ampl.param["TraineeExpiration"] = inst.trainee_expiration

    # TraineePreferences: 0 - not wanted, larger value <=> higher preference
    df = pd.DataFrame.from_dict(
        inst.trainee_preferences,
        orient="index",
    ).reindex(inst.trainees)
    ampl.param["TraineePreferences"] = df

    # Position capacity
    ampl.param["PositionCapacity"] = inst.position_capacity

    # Aggregated capacities (All, CK, CB)
    ampl.param["GroupCapacity"] = inst.group_capacity
    return ampl


def check_seniority_constraints(ampl: AMPL, inst: Instance):
    trainees, sessions = inst.trainees, inst.sessions
    preferences = {
        t: {sessions.index(s) for s in inst.trainee_preferences[t]} for t in trainees
    }
    position = inst.trainee_position
    seniority = inst.trainee_seniority
    language = inst.trainee_language
    expiration = inst.trainee_expiration
    preferences = inst.trainee_preferences
    position_capacity = inst.position_capacity
    group_capacity = inst.group_capacity
    position_groups = inst.position_groups
    # param G {t in Trainees} := last(PositionGroups[P[i]]);      # Group: CK or CB
    group = {t: position_groups[position[t]][1] for t in trainees}

    assign = ampl.get_variable("Assign").to_dict()
    unassigned = ampl.get_data(
        "{t in Trainees} (1- sum {s in Sessions} Assign[t, s]);"
    ).to_dict()
    session_language = ampl.get_variable("SessionLanguage").to_dict()

    num_violations, result_msg = 0, ""
    for t in trainees:
        for s in preferences[t]:
            if (
                expiration[t] > 0
                and preferences[t][s] == 0
                and round(unassigned[t]) == 1
                and round(session_language[s]) + 1 != language[t]
            ):  ## != means compatible language
                dominated_same_position = {
                    j
                    for j in trainees  ## Dominated trainees with same position
                    if (
                        position[j] == position[t]
                        and s in preferences[j]
                        and preferences[j][s] == 0
                        and (
                            (
                                expiration[j] == expiration[t]
                                and seniority[j] < seniority[t]
                            )
                            or expiration[j] > expiration[t]
                        )
                    )
                }
                dominated_same_group_diff_position = {
                    j
                    for j in trainees  ## Dominated trainees in the same group but diff position
                    if (
                        position[j] != position[t]
                        and group[j] == group[t]
                        and s in preferences[j]
                        and preferences[j][s] == 0
                        and (
                            (
                                expiration[j] == expiration[t]
                                and seniority[j] < seniority[t]
                            )
                            or expiration[j] > expiration[t]
                        )
                    )
                }
                dominated_other_groups = {
                    j
                    for j in trainees  ## Dominated trainees in the other group
                    if (
                        group[j] != group[t]
                        and s in preferences[j]
                        and preferences[j][s] == 0
                        and (
                            (
                                expiration[j] == expiration[t]
                                and seniority[j] < seniority[t]
                            )
                            or expiration[j] > expiration[t]
                        )
                    )
                }
                ## Now: dominating sets
                dominating_same_position = {
                    j
                    for j in trainees  ## Dominating trainees with same position
                    if (
                        position[j] == position[t]
                        and s in preferences[j]
                        and (
                            expiration[j] == 0
                            or (
                                expiration[j] == expiration[t]
                                and (
                                    seniority[j] > seniority[t] or preferences[j][s] > 0
                                )
                            )
                            or (expiration[j] > expiration[t] and preferences[j][s] > 0)
                        )
                    )
                }
                dominating_same_group = {
                    j
                    for j in trainees  ## Dominating trainees in the same group
                    if (
                        group[j] == group[t]
                        and s in preferences[j]
                        and (
                            expiration[j] == 0
                            or (
                                expiration[j] == expiration[t]
                                and (
                                    seniority[j] > seniority[t] or preferences[j][s] > 0
                                )
                            )
                            or (expiration[j] > expiration[t] and preferences[j][s] > 0)
                        )
                    )
                }
                at_position_capacity = (
                    round(sum([assign[j, s] for j in dominating_same_position]))
                    >= position_capacity[position[t]]
                )
                at_group_capacity = (
                    round(sum([assign[j, s] for j in dominating_same_group]))
                    >= group_capacity[group[t]]
                )
                dominated = [
                    dominated_same_position,
                    dominated_same_group_diff_position,
                    dominated_other_groups,
                ]
                dominating = [
                    [],
                    [dominating_same_position],
                    [dominating_same_position, dominating_same_group],
                ]
                at_limit = [
                    [],
                    [at_position_capacity],
                    [at_position_capacity, at_group_capacity],
                ]
                for k in range(3):
                    if round(sum([assign[j, s] for j in dominated[k]])) >= 1 and (
                        len(at_limit[k]) == 0 or max(at_limit[k]) == 0
                    ):
                        num_violations += 1
                        result_msg = (
                            result_msg
                            + "\n    RevSen VIOLATION: kind={}, [i, t]=[{}, {}], J={}, DJ={}, flgD={}".format(
                                k + 1,
                                t,
                                s,
                                dominated[k],
                                dominating[k],
                                at_limit[k],
                            )
                        )

    result_msg = (
        result_msg
        + "\n     ---- "
        + str(num_violations)
        + " original reverse seniority violations (with language compatibility)."
    )
    return num_violations == 0, result_msg


class SolveStats:
    def __init__(self):
        self.solve_results = []
        self.times = []
        self.reverse_seniority_violations = []
        self.average_preference_violations = []
        self.violation_balances = []

    def add(self, ampl: AMPL, inst: Instance):
        self.solve_results.append(ampl.solve_result)
        pv_ranked = ampl.obj["PreferenceViolationRanked"].to_pandas()
        pref_violation_average = pv_ranked.sum().sum() / len(pv_ranked.index)
        self.average_preference_violations.append(pref_violation_average)
        status, msg = check_seniority_constraints(ampl, inst)
        self.reverse_seniority_violations.append(not status)
        self.violation_balances.append(
            ampl.get_value(
                """
                sum {s in Sessions} abs( 
                    sum {t in Trainees: s in TraineeSessions[t]} Assign[t, s] 
                    - sum {t in Trainees, s1 in TraineeSessions[t]} Assign[t, s1] / card(Sessions) 
                ) / card(Sessions)
                """
            )
        )
        return msg, pv_ranked


def present_solution(ampl: AMPL, inst: Instance):
    assignments = dict(
        ampl.get_data("{t in Trainees, s in Sessions: Assign[t, s] > 0.5} Assign[t, s]")
        .to_dict()
        .keys()
    )
    st.write("## Optimal solution")
    st.write(
        pd.DataFrame(
            {
                "Assigned": {t: assignments.get(t, None) for t in assignments},
                "Desired": {
                    t: split_preferences(inst.trainee_preferences[t])[0]
                    for t in inst.trainees
                },
                "Undesired": {
                    t: split_preferences(inst.trainee_preferences[t])[1]
                    for t in inst.trainees
                },
            }
        ).reindex(inst.trainees)
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        stats = SolveStats()
        if ampl.solve_result != "solved":
            print("        !!!!!! solve_result NOT OPTIMAL.")
        msg, pv_ranked = stats.add(ampl, inst)
        average_preference_violation = stats.average_preference_violations[0]
        print(
            "AVERAGE NORMALIZED PREFERENCE VIOLATION = {:.4}".format(
                average_preference_violation
            )
        )
        # Seniority as objectives
        print("REVERSE SENIORITY:")
        print("   ---->> AS PBS OBJECTIVES:")
        print(
            ampl.get_output(
                """
                display {e in 1..2}
                    sum {t in Trainees, s in Sessions: 
                            TraineeExpiration[t] == e
                            and s in TraineeSessions[t]
                            and TraineePreferences[t, s] == 0}
                        TraineeSeniority[t] * Assign[t, s];
                """
            )
        )
        print(
            "   ---->> AS DOMINANCE CONSTRAINTS FROM [1]: {}; {}".format(
                "Violated" if stats.reverse_seniority_violations[0] else "OK", msg
            )
        )
        print(
            "AVERAGE SESSION LOAD IMBALANCE = {:.3f}\n".format(
                stats.violation_balances[0]
            )
        )

    st.write(f"```\n{output.getvalue()}\n```")

    for name, obj in ampl.get_objectives():
        st.write(f"Objective {name}:")
        df = obj.to_pandas()
        if name == "PreferenceViolationRanked":
            df.index.names = ["Seniority"]
        elif name == "ReverseSeniority":
            df = df.reindex(inst.trainees, level="index1")
            df.index.names = ["Expiration", "Trainee"]
        st.write(df)

    # Preferences
    plot = pv_ranked.plot(
        title="Ranked normalized preference violations", xlabel="Seniority"
    )
    plot.hlines(
        average_preference_violation,
        0,
        len(pv_ranked.index),
        label="Average",
        linestyle="--",
        color="pink",
    )
    plt.legend(("PreferenceViolationRanked", "Average"))
    st.pyplot(plt)
    # Solution as a heat map
    assign_df = ampl.get_variable("Assign").to_pandas().unstack().T
    plt.clf()
    plt.imshow(assign_df)
    plt.title("Schedule")
    plt.xlabel("Trainees")
    plt.ylabel("Classes")
    st.pyplot(plt)
    # Free capacity
    cap = ampl.get_data("PositionCapacityLimit.slack").to_pandas().unstack().T
    cap.columns = inst.positions
    cap.index = inst.sessions
    plot = cap.plot(
        title="Free capacity by position", xlabel="Class slot", kind="bar", stacked=True
    )
    st.pyplot(plt)
    # Free group capacity
    cap = ampl.get_data("GroupCapacityLimit.slack").to_pandas().unstack().T
    cap.columns = inst.meta_positions
    cap.index = inst.sessions
    plot = cap.plot(
        title="Free group capacity", xlabel="Class slot", kind="bar", stacked=True
    )
    st.pyplot(plt)
    # Session language
    st.write("Session language")
    df = (
        ampl.get_variable("SessionLanguage")
        .to_pandas()
        .T.reindex(columns=inst.sessions)
    )
    st.write(df)
    # Minimal value of the logical constraints (1 when all true)
    assert 1 == ampl.get_data("Language1").to_pandas()["Language1"].min()
    assert 1 == ampl.get_data("Language2").to_pandas()["Language2"].min()


def main():
    st.header("✈️ Aircrew training scheduling with seniority constraints")

    st.markdown(
        """
        This app considers a realistic trainee scheduling model from [1] involving classroom capacities, language and (reverse) seniority constraints, and seniority-ranked preferences. The (compulsory) seniority constraints are modeled simply and efficiently by Preferential Bidding System (PBS)-style secondary objective functions.
        Learn more with our notebook on Google Colab: [Aircrew trainee scheduling](https://ampl.com/colab/tags/aircrew-scheduling.html).
        """
    )

    with st.expander(
        "📁 AMPL model for aircrew training scheduling with seniority constraints",
        expanded=True,
    ):
        st.code(
            open(os.path.join(os.path.dirname(__file__), "airtrainee.mod"), "r").read()
        )

    rng = np.random.default_rng(1234)

    left, right = st.columns(2)
    with left:
        # With the Demo license, set up to 50
        num_trainees = st.slider(
            "Number of trainees 👇", min_value=25, max_value=200, step=1, value=50
        )
    with right:
        min_value = max(5, int(math.ceil(num_trainees / 13)))
        num_sessions = st.slider(
            "Number of sessions 👇",
            min_value=min_value,
            max_value=num_trainees,
            step=1,
            value=min_value * 2,
        )

    generator = InstanceGenerator(num_trainees, num_sessions, rng)
    with st.expander("Instance generation configuration"):
        generator.generator_editor()
    instance = generator.generate_instance()
    instance.instance_editor()

    # with st.expander("JSON Instance"):
    #     st.write(instance.to_dict())
    # json_data = instance.to_json()
    # open(os.path.join(os.path.dirname(__file__), "input.json"), "w").write(json_data)

    ampl = make_ampl_instance(["airtrainee.mod"], instance)
    load_imbalance = st.checkbox(
        "Minimize session load imbalance?",
        help="""
            Similar to the last model extension in [1], we post-process the solution
            to minimize session load imbalance. We only tackle the imbalance in overall class loads.
            For that we add yet another objective function.
        """,
    )
    if not load_imbalance:
        ampl.obj["LoadImbalance"].drop()

    solvers = ["highs", "scip", "cbc", "gurobi", "xpress", "cplex", "mosek", "copt"]
    solver = st.selectbox("Pick the solver to use 👇", solvers, key="solver")
    if solver == "cplex":
        solver = "cplexmp"

    output = ampl.solve(
        solver=solver,
        mp_options="outlev=1 multiobj=1 tech:timing=1",  # mp_options for MP-based solvers
        gurobi_options="mip:intfocus=1",
        return_output=True,
    )
    with st.expander("📄 Solve process output"):
        st.write(f"```\n{output}\n```")
    st.write(
        f"Solver: {solver}, Solve result: {ampl.solve_result}, Time: {float(ampl.get_value('_total_solve_time')):.3}s"
    )

    if ampl.solve_result == "solved":
        present_solution(ampl, instance)

    st.markdown(
        """
        ## References:
        1. Kozanidis, G. (2017). Optimal assignment of aircrew trainees to simulator and classroom training sessions subject to seniority and preference restrictions. Journal of Air Transport Management 59, 143-154.
        2. Gamache, M., Soumis, F., Villeneuve, D., Desrosiers, J., & Gélinas, É. (1998). The preferential bidding system at Air Canada. Transportation Science, 32(3), 246-255.
        3. Achour, H., Gamache, M., Soumis, F., & Desaulniers, MetaPositions. (2007). An exact solution approach for the preferential bidding system problem in the airline industry. Transportation Science, 41(3), 354-365.
        """
    )

    st.markdown(
        """
    #### [[App Source Code on GitHub](https://github.com/fdabrandao/amplopt.streamlit.app/tree/master/apps/aircrew_training_scheduling)]"""
    )
