import torch
import numpy as np
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt

temp = torch.Tensor([])
time = torch.Tensor([])

BOUNDS_temp = (150.0, 200.0)
BOUNDS_time = (8.0, 15.0)

variables = [("Temperature", "Baking time")]
BOUNDS = torch.tensor([BOUNDS_temp, BOUNDS_time])
D = len(variables)
Q=3

ROUNDS = 2
SEED = 12345
torch.manual_seed(seed=SEED)
bounds_tensor = torch.tensor(BOUNDS, dtype=torch.float64).T  

#Gaussian like peak as the objective function
def cookie_quality(variables):
    temp, time = x[..., 0], x[..., 1]
    return np.exp(-0.01 * (temp - 200)**2 - 0.1 * (time - 10)**2) * 10


#Initial data
torch.set_default_dtype(torch.float64)
x = draw_sobol_samples(bounds=bounds_tensor, n=1,q=Q, seed=SEED).squeeze(0)
y = cookie_quality(x).view(-1, 1)  

#Surrogate model
gp_model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        input_transform=Normalize(d=x.shape[1]),
        outcome_transform=Standardize(m=1),
        )
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)

best_kpi_values=[]

#acqisition function
sampler=SobolQMCNormalSampler(torch.Size([Q]),seed=SEED)
qlei = qLogExpectedImprovement(
        model=gp_model,
        best_f=y.max(),
        sampler=sampler,
        )

#iteration loop
for round_num in range(ROUNDS):
    print(f"Round {round_num + 1} optimization:")
    # Optimize the acquisition function to find the next batch of experiments
    next_x, _ = optimize_acqf(
        acq_function=qlei,
        bounds=bounds_tensor,
        q=Q,
        num_restarts=20,
        raw_samples=100,
    )

    next_y = torch.concat([cookie_quality(xi) for xi in next_x], dim=0).view(-1,1)

    # Update the dataset with the new data
    x = torch.concat([x, next_x], dim=0)
    y = torch.concat([y, next_y], dim=0)

    
    # Track the best KPI value (the highest EFE_m production) for this round
    best_kpi_values.append(y.max().item())

last_x=x[-1]
print(f"Best temperature and time for baking:", last_x)

#Representation of Improvement over the rounds
plt.plot(range(1, len(best_kpi_values) + 1), best_kpi_values, marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.title('Improvement of Cookies Quality Over Optimization Rounds', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Optimization Round', fontsize=12, color='black')
plt.ylabel('Cookies Quality', fontsize=12, color='black')
plt.gcf().set_facecolor('whitesmoke')
plt.show()