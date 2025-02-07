import torch
from bayesian_functions import get_test_function, bayesian_optimization, plot_kpi_progress
from cobra.io import read_sbml_model
import bayesian_functions
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import Ackley
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.synthetic import Hartmann
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iJO1366.xml')

#Define the Search Space
MEDIA=model.medium
BOUNDS = [(0.0, max_value) for max_value in MEDIA.values()]
Q=12
D=len(MEDIA)
ROUNDS = 5
SEED = 12345
torch.manual_seed(SEED)

#Define objective function
def compute_growth_rate(media_composition):
    for i, key in enumerate(MEDIA.keys()):
        model.medium[key] = float(media_composition[i].item())

    solution = model.optimize()
    return torch.tensor([[solution.objective_value]])

#Train the surrogate model
bounds_tensor = torch.Tensor(BOUNDS).T
x = draw_sobol_samples(bounds=bounds_tensor, q=Q, n=1, seed=SEED).squeeze(0)
y = torch.cat([compute_growth_rate(xi) for xi in x])
gp_model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        input_transform=Normalize(d=x.shape[1]),
        outcome_transform=Standardize(m=1),
        )
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)





