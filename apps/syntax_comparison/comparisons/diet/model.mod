# @block sets
set NUTR;  # nutrients
set FOOD;  # foods that can be bought

# @block params
param cost {FOOD} > 0;
param f_min {FOOD} >= 0;
param f_max {j in FOOD} >= f_min[j];
param n_min {NUTR} >= 0;
param n_max {i in NUTR} >= n_min[i];
param amt {NUTR, FOOD} >= 0;  # amount of nutrient i in one unit of food j

# @block vars
var Buy {j in FOOD} >= f_min[j], <= f_max[j];

# @block objective
minimize Total_Cost: sum {j in FOOD} cost[j] * Buy[j];

# @block constraints
subject to Diet {i in NUTR}:
    n_min[i] <= sum {j in FOOD} amt[i,j] * Buy[j] <= n_max[i];
