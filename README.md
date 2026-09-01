# exp 6
import numpy as np

# 1. Setup Grid Dimensions
rows, cols = 3, 4

# 2. Initialize Utilities and Rewards
# To map the Cartesian coordinates (row, col) nicely: 
# (3,4) corresponds to array index [2, 3] (top-right)
# (2,2) corresponds to array index [1, 1] (center-left)
# (2,4) corresponds to array index [1, 3] (center-right)
U = np.zeros((rows, cols))
R = np.full((rows, cols), -0.04)

R[2, 3] = 1.0   # Goal (3,4)
#R[1, 1] = -1.0  # Pit (2,2)
R[1, 3] = -1.0  # Penalty (2,4)

terminals = [(2,3), (1,3)]

# Preset fixed utility values for terminal states
for r, c in terminals:
    U[r, c] = R[r, c]

actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# 3. Define Movement Rules (Handling Boundaries & Bumping)
def get_next_state(r, c, action):
    if action == 'UP':    next_r, next_c = r + 1, c
    elif action == 'DOWN': next_r, next_c = r - 1, c
    elif action == 'LEFT': next_r, next_c = r, c - 1
    elif action == 'RIGHT': next_r, next_c = r, c + 1
    
    if 0 <= next_r < rows and 0 <= next_c < cols:
        if next_r==1 and next_c==1:
            return r,c
        return next_r, next_c
    return r, c  # Agent bumps into wall and stays in place

# Define 90-degree drift probability offsets
def get_action_distribution(action):
    if action == 'UP':    return 'UP', 'LEFT', 'RIGHT'
    if action == 'DOWN':  return 'DOWN', 'RIGHT', 'LEFT'
    if action == 'LEFT':  return 'LEFT', 'DOWN', 'UP'
    if action == 'RIGHT': return 'RIGHT', 'UP', 'DOWN'

# 4. Main Value Iteration Loop
gamma = 1.0
epsilon = 1e-4

while True:
    U_next = np.copy(U)
    delta = 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) in terminals:
                continue
            
            action_values = []
            for a in actions:
                intended, left, right = get_action_distribution(a)
                
                sr_i, sc_i = get_next_state(r, c, intended)
                sr_l, sc_l = get_next_state(r, c, left)
                sr_r, sc_r = get_next_state(r, c, right)
                
                # Bellman Expectation
                expected_utility = (0.8 * U[sr_i, sc_i] + 
                                    0.1 * U[sr_l, sc_l] + 
                                    0.1 * U[sr_r, sc_r])
                action_values.append(expected_utility)
                
            U_next[r, c] = R[r, c] + gamma * max(action_values)
            delta = max(delta, abs(U_next[r, c] - U[r, c]))
            
    U = U_next
    if delta < epsilon:  # Threshold for convergence met
        break

# 5. Extract Optimal Policy
policy = {}
for r in range(rows):
    for c in range(cols):
        if (r, c) in terminals:
            policy[(r, c)] = 'GOAL' if (r, c) == (2,3) else 'TRAP'
            continue
        best_action = None
        best_val = -float('inf')
        for a in actions:
            intended, left, right = get_action_distribution(a)
            sr_i, sc_i = get_next_state(r, c, intended)
            sr_l, sc_l = get_next_state(r, c, left)
            sr_r, sc_r = get_next_state(r, c, right)
            val = 0.8 * U[sr_i, sc_i] + 0.1 * U[sr_l, sc_l] + 0.1 * U[sr_r, sc_r]
            if val > best_val:
                best_val = val
                best_action = a
        policy[(r, c)] = best_action

# 6. Format Outputs to Align with a visual 3x4 Grid Layout
print("--- Final Utility Table ---")
print(np.round(np.flipud(U), 3))

print("\n--- Extracted Policy Layout ---")
p_grid = np.empty((rows, cols), dtype=object)
for r in range(rows):
    for c in range(cols):
        p_grid[r, c] = policy[(r, c)]
print(np.flipud(p_grid))
