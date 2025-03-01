import torch
from cobra.io import read_sbml_model
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

#__import__("pdb").set_trace()

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/modified_model.xml')

#Define the Search Space
MEDIA=model.medium
BOUNDS = [(0.0, max_value) for max_value in MEDIA.values()]
Q=12
D=len(MEDIA)
ROUNDS = 3
SEED = 12345
torch.manual_seed(SEED)

#convert tensor to a media dictionary for the model
def tensor_to_media_dict(tensor, media_template):
    media_keys = list(media_template.keys())
    return {media_keys[i]: float(tensor[i].item()) for i in range(len(media_keys))}

#Define objective function
def compute_ethylene_production(media_tensor):
    media_composition = tensor_to_media_dict(media_tensor, model.medium)
    with model:
        model.medium = media_composition  # Set the medium
        model.objective = model.reactions.EFE_m  # Set the objective
        solution = model.optimize()  # Optimize the model
        ethylene_production = solution.objective_value  # Extract ethylene production flux
    
    # Return the ethylene production as a PyTorch tensor
    return torch.tensor([[ethylene_production]], dtype=torch.float64)

#Train the surrogate model
torch.set_default_dtype(torch.float64)
bounds_tensor = torch.Tensor(BOUNDS).T
x = draw_sobol_samples(bounds=bounds_tensor, q=Q, n=1, seed=SEED).squeeze(0)
y = torch.Tensor([])
y = torch.concat([compute_ethylene_production(xi) for xi in x], dim=0).view(-1,1)

gp_model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        input_transform=Normalize(d=x.shape[1]),
        outcome_transform=Standardize(m=1),
        )
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)

best_kpi_values=[]
#Define and Optimize acquisition function
sampler=SobolQMCNormalSampler(torch.Size([Q]),seed=SEED)
qlei = qLogExpectedImprovement(
        model=gp_model,
        best_f=y.max(),
        sampler=sampler,
        )


#Update data with the new rounds
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
    
    # Run new experiments and calculate the corresponding y values (KPI values)
    next_y = torch.concat([compute_ethylene_production(xi) for xi in next_x], dim=0).view(-1,1)

    # Update the dataset with the new data
    x = torch.concat([x, next_x], dim=0)
    y = torch.concat([y, next_y], dim=0)
    
    # Track the best KPI value (the highest EFE_m production) for this round
    best_kpi_values.append(y.max().item())

print(x)
print(y)


# Plot the improvement of the KPI over the rounds
plt.plot(range(1, len(best_kpi_values) + 1), best_kpi_values, marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.title('Improvement of EFE_m Production Over Optimization Rounds', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Optimization Round', fontsize=12, color='black')
plt.ylabel('Best EFE_m Production (KPI)', fontsize=12, color='black')
plt.gcf().set_facecolor('whitesmoke')
plt.show()