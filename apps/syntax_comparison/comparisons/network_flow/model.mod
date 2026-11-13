# @block sets
set CITIES;
set PRODS;
set LINKS within {CITIES, CITIES};  # shipping lanes (orig, dest)
set ROUTES within {LINKS, PRODS};  # lanes each product may use (e.g. Frozen needs refrigeration)

# @block params
param supply {CITIES, PRODS} default 0;  # > 0 at plants, < 0 at markets, 0 elsewhere
param cost {ROUTES} >= 0;  # cost per unit shipped
param capacity {LINKS} >= 0;  # shared by all products on the lane

# @block vars
var Ship {ROUTES} >= 0;

# @block objective
minimize Total_Cost: sum {(i,j,p) in ROUTES} cost[i,j,p] * Ship[i,j,p];

# @block constraints
subject to Balance {k in CITIES, p in PRODS}:
    supply[k,p] + sum {(i,k,p) in ROUTES} Ship[i,k,p] = sum {(k,j,p) in ROUTES} Ship[k,j,p];

subject to Capacity {(i,j) in LINKS}:
    sum {(i,j,p) in ROUTES} Ship[i,j,p] <= capacity[i,j];
