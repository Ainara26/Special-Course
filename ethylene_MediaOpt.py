import torch
from bayesian_functions import get_test_function, bayesian_optimization, plot_kpi_progress
from cobra.io import read_sbml_model

import bayesian_functions
model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iJO1366.xml')

MEDIA=model.medium
BOUNDS = []
for exchange, max_value in MEDIA.items():
    BOUNDS.append((0.0, max_value))
Q=12
D=len(MEDIA)
SEED = 12345
ROUNDS = 5
# Step 1: Select the right test function
TEST_FUNCTION = get_test_function(D, BOUNDS)

# Step 2: Generate Initial Data
bounds_tensor = torch.Tensor(BOUNDS).T
x_init = torch.rand(Q, D)  # Random initial conditions
y_init = TEST_FUNCTION.forward(x_init, noise=True).unsqueeze(-1) if TEST_FUNCTION else torch.rand(Q, 1)

# Step 3: Run Bayesian Optimization
x_final, y_final, best_kpis = bayesian_optimization(x_init, y_init, BOUNDS, Q, ROUNDS, SEED, TEST_FUNCTION)

# Step 4: Plot KPI Progress
plot_kpi_progress(best_kpis)

print("Selected Test Function:", TEST_FUNCTION)




