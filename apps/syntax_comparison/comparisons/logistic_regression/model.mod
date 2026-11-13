# @block sets
set POINTS;  # applicants
set FEATURES;  # standardized features

# @block params
param x {POINTS, FEATURES};
param y {POINTS} binary;  # 1 = approved
param lambda >= 0;  # regularization weight

# @block vars
var w {FEATURES};
var b;

# @block objective
minimize Log_Loss:
    sum {i in POINTS} log(1 + exp(-(2*y[i] - 1) * (sum {f in FEATURES} w[f] * x[i,f] + b)))
  + lambda * sum {f in FEATURES} w[f]^2;
