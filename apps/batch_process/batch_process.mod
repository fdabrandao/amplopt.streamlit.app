set TASKS;
set UNITS;
set TIME;
param H;  # End time or horizon

set STATES;
param price{STATES};
param initial{STATES};

set I{j in UNITS};  # I[j] set of tasks performed with unit j
set K{i in TASKS};  # K[i] set of units capable of task i

set S_In{i in TASKS};  # S_In[i] input set of states which feed task i
set S_Out{i in TASKS};  # S_Out[i] output set of states fed by task i
set T_In{s in STATES};  # T_In[s] set of tasks producing material for state s
set T_Out{s in STATES};  # T_Out[s] set of tasks receiving material from state s

param P{i in TASKS, s in S_Out[i]};  # P[i,s] time for task i output to state s
param p{i in TASKS};  # Processing times for tasks
param Tclean{(j,i) in UNITS cross TASKS};

param C{s in STATES};  # Capacity constraints for states

param rho_in{i in TASKS, s in S_In[i]};  # rho_in[i,s] input fraction of task i from state s
param rho_out{i in TASKS, s in S_Out[i]};  # rho_out[i,s] output fraction of task i to state s

param Bmin{(i,j) in UNITS cross TASKS};
param Bmax{(i,j) in UNITS cross TASKS};
param Cost{(i,j) in UNITS cross TASKS} >= 0;
param vCost{(i,j) in UNITS cross TASKS} >= 0;

# W[i,j,t] 1 if task i starts in unit j at time t
var W{i in TASKS, j in K[i], t in TIME} binary;

# B[i,j,t] size of batch assigned to task i in unit j at time t
var B{i in TASKS, j in K[i], t in TIME} >= 0;

# S[s,t] inventory of state s at time t
var S{s in STATES, t in TIME} >= 0, <= C[s];

# Q[j,t] inventory of unit j at time t
var Q{j in UNITS, t in TIME} >= 0;

# Objective function
var TotalValue = sum{s in STATES} price[s] * S[s,H];
var TotalCost = sum{i in TASKS, j in K[i], t in TIME} (Cost[j,i] * W[i,j,t] + vCost[j,i] * B[i,j,t]);;

# Objective function
maximize Total_Profit: TotalValue - TotalCost;

# Constraints

# a unit can only be allocated to one task
subject to Unit_Allocation{j in UNITS, t in TIME}:
    sum{i in I[j], tprime in TIME: tprime >= (t - p[i] + 1 - Tclean[j,i]) && tprime <= t} W[i,j,tprime] <= 1;

# state mass balances
subject to State_Balance{s in STATES, t in TIME}:
    S[s,t] = (if t > 0 then S[s,t-1] else initial[s])
           + sum {i in T_In[s], j in K[i]: t >= P[i,s]} rho_out[i,s]*B[i,j,t-P[i,s]]
           - sum {i in T_Out[s], j in K[i]} rho_in[i,s] * B[i,j,t];

# unit capacity constraints
subject to Unit_Capacity_Min{i in TASKS, j in K[i], t in TIME}:
  W[i,j,t]*Bmin[j,i] <= B[i,j,t];

subject to Unit_Capacity_Max{i in TASKS, j in K[i], t in TIME}:
  B[i,j,t] <= W[i,j,t]*Bmax[j,i];

# unit mass balances
subject to Unit_Balance{j in UNITS, t in TIME}:
    Q[j,t] = (if t > 0 then Q[j,t-1] else 0)
           + sum {i in I[j]} B[i,j,t]
           - sum {i in I[j], s in S_Out[i]: t >= P[i,s]} rho_out[i,s] * B[i,j,t-P[i,s]];

# unit terminal condition
subject to Terminal_Condition{j in UNITS}:
    Q[j,H] = 0;