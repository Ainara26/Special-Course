import torch
from bayesian_functions import get_test_function, bayesian_optimization, plot_kpi_progress
from cobra.io import read_sbml_model

import bayesian_functions
model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iJO1366.xml')

MEDIA=model.medium
BOUNDS=[(0.0, 1000.0)] * len(MEDIA)
Q=12
D=len(MEDIA)
TRUTH = Ackley(dim=D, bounds=BOUNDS, negate=True, noise_std=0.05)
TRUE_ARGOPTIMUM = (40, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 20, 20, 20, 20, 0, 0, 0, 20)
TRUE_OPTIMUM = 17.292327443315337
SEED = 12345
torch.manual_seed(seed=SEED)
exec bayesian_functions.py 



