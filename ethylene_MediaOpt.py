import torch
from cobra.io import read_sbml_model
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

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/modified_model.xml')

#Define the Search Space
MEDIA=model.medium
#BOUNDS=[(0.0, 1.0)] * len(MEDIA) 
BOUNDS = [(0.0, max_value) for max_value in MEDIA.values()]
Q=12
D=len(MEDIA)
ROUNDS = 2
SEED = 12345
torch.manual_seed(SEED)

#Define objective function
def compute_ethylene_production(media_composition):
    with model:
        solution=model.optimize()
        model.objective=model.reactions.EFE_m
        E_production = solution.objective_value
    return torch.tensor([[E_production]])

#Train the surrogate model
bounds_tensor = torch.Tensor(BOUNDS).T
x = draw_sobol_samples(bounds=bounds_tensor, q=Q, n=1, seed=SEED).squeeze(0)
y = torch.cat([compute_ethylene_production(xi) for xi in x], dim=0)
gp_model = SingleTaskGP(
        train_X=x,
        train_Y=y.squeeze(-1),
        input_transform=Normalize(d=x.shape[1]),
        outcome_transform=Standardize(m=1),
        )
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)

best_kpi_values=[]
#Define and Optimize acquisition function
    #define
sampler=SobolQMCNormalSampler(torch.Size([Q]),seed=SEED) 
qlei=qLogExpectedImprovement(model=gp_model, best_f=float(y.max().item()), sampler=sampler)
    #optimize to find the next batch of experiments

#Update data with the new rounds
for round_num in range(ROUNDS):
    print(f"Round {round_num + 1} optimization:")
    #__import__("pdb").set_trace()
    # Optimize the acquisition function to find the next batch of experiments
    next_x, _ = optimize_acqf(
        acq_function=qlei,
        bounds=bounds_tensor,
        q=Q,
        num_restarts=10,
        raw_samples=50,
    )
    
    # Run new experiments and calculate the corresponding y values (KPI values)
    next_y = torch.cat([compute_ethylene_production(xi).unsqueeze(-1) for xi in next_x], dim=0)


    # Update the dataset with the new data
    x = torch.cat([x, next_x], dim=0)
    y = torch.cat([y, next_y], dim=0)

    # Retrain the GP model with new data
    gp_model.set_train_data(x, y, strict=False)
    fit_gpytorch_mll(mll)
    
    # Track the best KPI value (the highest EFE_m production) for this round
    best_kpi_values.append(y.max().item())

# Plot the improvement of the KPI over the rounds
plt.plot(range(1, ROUNDS + 1), best_kpi_values, marker='o')
plt.xlabel('Optimization Round')
plt.ylabel('Best EFE_m Production (KPI)')
plt.title('Improvement of EFE_m Production Over Optimization Rounds')
plt.grid(True)
plt.show()



