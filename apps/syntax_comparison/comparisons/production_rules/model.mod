# @block sets
set PROD;                               # products made on the line
set PERIODS ordered;                    # planning weeks
set CONFLICTS within {PROD, PROD};      # pairs that cannot share a week (allergen cleaning)

# @block params
param demand {PROD, PERIODS} >= 0;
param hours {PERIODS} >= 0;             # line hours available each week
param rate {PROD} > 0;                  # units produced per line hour
param min_lot {PROD} > 0;               # smallest batch worth running
param max_lot {p in PROD} >= min_lot[p];
param make_cost {PROD} >= 0;
param hold_cost {PROD} >= 0;
param init_inv {PROD} >= 0;
param max_inv {PROD} >= 0;
param reorder {PROD} >= 0;              # reorder point
param max_products integer > 0;         # changeover limit per week
param smooth_cost >= 0;                 # cost per unit of week-to-week change
param peak_cost >= 0;                   # cost per hour of the busiest week

# @block vars
var Make {p in PROD, PERIODS} >= 0, <= max_lot[p];
var Inv {p in PROD, PERIODS} >= 0, <= max_inv[p];

# @block objective
minimize Total_Cost:
    sum {p in PROD, t in PERIODS} (make_cost[p] * Make[p,t] + hold_cost[p] * Inv[p,t])
  + smooth_cost * sum {p in PROD, t in PERIODS: ord(t) > 1} abs(Make[p,t] - Make[p,prev(t)])
  + peak_cost * max {t in PERIODS} sum {p in PROD} Make[p,t] / rate[p];

# @block constraints
subject to Balance {p in PROD, t in PERIODS}:
    Inv[p,t] = (if ord(t) = 1 then init_inv[p] else Inv[p,prev(t)]) + Make[p,t] - demand[p,t];

subject to Capacity {t in PERIODS}:
    sum {p in PROD} Make[p,t] / rate[p] <= hours[t];

subject to Min_Lot {p in PROD, t in PERIODS}:
    Make[p,t] = 0 or Make[p,t] >= min_lot[p];

subject to Max_Products {t in PERIODS}:
    count {p in PROD} (Make[p,t] > 0) <= max_products;

subject to No_Mix {(p,q) in CONFLICTS, t in PERIODS}:
    Make[p,t] = 0 or Make[q,t] = 0;

subject to Reorder {p in PROD, t in PERIODS: ord(t) < card(PERIODS)}:
    Inv[p,t] < reorder[p] ==> Make[p,next(t)] > 0;
