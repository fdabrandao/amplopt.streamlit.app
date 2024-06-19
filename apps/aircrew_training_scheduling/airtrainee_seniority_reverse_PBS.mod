
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