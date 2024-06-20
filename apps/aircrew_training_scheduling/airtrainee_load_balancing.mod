# Minimize overall session load imbalance
minimize LoadImbalance:
  sum {s in Sessions}
    abs(
      sum {t in Trainees: s in TraineeSessions[t]} Assign[t, s] 
      - sum {t in Trainees, s1 in TraineeSessions[t]} Assign[t, s1] / card(Sessions)
    )
  suffix objpriority 0;
