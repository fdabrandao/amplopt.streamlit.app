set Trainees;                 # Trainees
set Sessions;                 # Training sessions
set Positions;                # Positions
set MetaPositions;            # Meta-positions (e.g, All, Cockpit, Cabin)
set PositionGroups{Positions} in MetaPositions ordered; # Groups corresponding to each position

param TraineePosition{Trainees} in Positions symbolic;  # Trainee's position
param TraineeSeniority{Trainees};             # Seniority (smaller value <=> higher seniority)
param TraineeLanguage{Trainees};              # Language (0 - both, 1 or 2 - one only)
param TraineeExpiration{Trainees};            # Expiration: 0 - this month, 1 - next month, 2 - in 2 months
param TraineePreferences{Trainees, Sessions} default -1;    # Priority: -1 -  not possible, 0 - not wanted, larger value <=> higher preference
param PositionCapacity{Positions};            # Position capacity
param GroupCapacity{MetaPositions};           # Aggregated capacities (All, Cockpit, Cabin)

set TraineeSessions{t in Trainees} in Sessions := {s in Sessions: TraineePreferences[t, s] >= 0}; # Valid sessions for each trainee

var Assign{t in Trainees, s in Sessions} binary <= if s in TraineeSessions[t] then 1 else 0;
var Unassigned{t in Trainees} binary <= if TraineeExpiration[t]>0 then 1 else 0;  # Trainee t unassigned
var SessionLanguage{Sessions} binary;    # 1 <=> language 1, 0 <=> language 2
var UnassignedCount = sum {t in Trainees: TraineeExpiration[t] > 0} Unassigned[t];   # Number of unassigned trainees

s.t. MustAssignIfExpiring {t in Trainees: TraineeExpiration[t] == 0}:
  sum {s in TraineeSessions[t]} Assign[t, s] == 1;

s.t. AllowSkippingIfNotExpiring {t in Trainees: TraineeExpiration[t] > 0}:
  sum {s in TraineeSessions[t]} Assign[t, s] + Unassigned[t] == 1;

s.t. Language1 {s in Sessions}:
  SessionLanguage[s] < 0.5 ==> sum {t in Trainees: TraineeLanguage[t]==1 and s in TraineeSessions[t]} Assign[t, s] <= 0;

s.t. Language2 {s in Sessions}:
  SessionLanguage[s] >= 0.5 ==> sum {t in Trainees: TraineeLanguage[t]==2 and s in TraineeSessions[t]} Assign[t, s] <= 0;

s.t. PositionCapacityLimit {p in Positions, s in Sessions}:
  sum {t in Trainees: p==TraineePosition[t] and s in TraineeSessions[t]} Assign[t, s] <= PositionCapacity[p];

s.t. GroupCapacityLimit {g in MetaPositions, s in Sessions}:
  sum {t in Trainees: g in PositionGroups[TraineePosition[t]] and s in TraineeSessions[t]} Assign[t, s] <= GroupCapacity[g];

suffix objpriority;
param SeniorityRange := max {t in Trainees} TraineeSeniority[t] - min {t in Trainees} TraineeSeniority[t];

# The primary objective
minimize TotalUnassigned: UnassignedCount suffix objpriority 3*SeniorityRange + 1;

# Trainee preferences, ranked by seniority
set SeniorityLevels := setof {t in Trainees} TraineeSeniority[t];
param MaxPriority {t in Trainees} := max {s in TraineeSessions[t]} TraineePreferences[t, s];

minimize PreferenceViolationRanked {l in SeniorityLevels}:
  sum {t in Trainees: TraineeSeniority[t] == l}
    (1.0                                   # penalty 1 for non-assignment
     - sum {s in TraineeSessions[t]: TraineePreferences[t, s] > 0}
        TraineePreferences[t, s] / MaxPriority[t]                           # normalize
          * Assign[t, s]
     + sum {s in TraineeSessions[t]: TraineePreferences[t, s] == 0} Assign[t, s])           # penalty 2 for unwanted assignment
  suffix objpriority max {t in Trainees} TraineeSeniority[t] + 2*SeniorityRange + 1 - l;

# Reverse seniority constraints,
# sublimated as post-processing objectives.
# Optimize for all trainees with TraineeExpiration[t]==1, ranked by reverse seniority,
# then for all with TraineeExpiration[t]==2.
# In this special case the objectives can be aggregated for each value of e
# (see Solve with aggregated preferences.)
maximize ReverseSeniority {e in 1..2, t in Trainees: TraineeExpiration[t] == e}:
  sum {s in TraineeSessions[t]: TraineePreferences[t, s]==0}
    TraineeSeniority[t] * Assign[t, s]
  suffix objpriority (2-e)*SeniorityRange + 1 + TraineeSeniority[t] - min {ti in Trainees} TraineeSeniority[ti];

# Minimize overall session load imbalance
minimize LoadImbalance:
  sum {s in Sessions}
    abs(
      sum {t in Trainees: s in TraineeSessions[t]} Assign[t, s] 
      - sum {t in Trainees, s1 in TraineeSessions[t]} Assign[t, s1] / card(Sessions)
    )
  suffix objpriority 0;
