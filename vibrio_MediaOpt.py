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

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iLC858.sbml')

#Define the Search Space
MEDIA = model.medium
MEDIA['EX_Na+_e'] = 20.0
MEDIA['EX_Acetate_e']=2.5
MEDIA['EX_NH3_e']=1.0
MEDIA['EX_K+_e']=0.2
MEDIA['EX_Mg_e']=0.2

BOUNDS = [(0.0, 10*value) for value in MEDIA.values()]

Q=12
D=len(MEDIA)
ROUNDS = 2
SEED = 12345
torch.manual_seed(SEED)

#convert tensor to a media dictionary for the model
def tensor_to_media_dict(tensor, media_template):
    media_keys = list(media_template.keys())
    return {media_keys[i]: float(tensor[i].item()) for i in range(len(media_keys))}

#Define objective function
def compute_max_growth(media_tensor):
    media_composition = tensor_to_media_dict(media_tensor, model.medium)
    with model:
        model.medium = media_composition  # Set the medium
        #automatically the objective reaction is set to be the biomass (growth rate)
        solution = model.optimize()  # Optimize the model
        max_theoretical_growth = solution.objective_value  # Extract max theoretical growth
    
    # Return the ethylene production as a PyTorch tensor
    return torch.tensor([[max_theoretical_growth]], dtype=torch.float64)

#Train the surrogate model
torch.set_default_dtype(torch.float64)
bounds_tensor = torch.Tensor(BOUNDS).T
x = draw_sobol_samples(bounds=bounds_tensor, q=Q, n=1, seed=SEED).squeeze(0)
y = torch.Tensor([])
y = torch.concat([compute_max_growth(xi) for xi in x], dim=0).view(-1,1)

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
    #define
sampler=SobolQMCNormalSampler(torch.Size([Q]),seed=SEED)
qlei = qLogExpectedImprovement(
        model=gp_model,
        best_f=y.max(),
        sampler=sampler,
        )

    #optimize to find the next batch of experiments


#__import__("pdb").set_trace()

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
    next_y = torch.concat([compute_max_growth(xi) for xi in next_x], dim=0).view(-1,1)

    # Update the dataset with the new data
    x = torch.concat([x, next_x], dim=0)
    y = torch.concat([y, next_y], dim=0)
    
    best_so_far = float(y.max())
    best_kpi_values.append(best_so_far)


#plot
plt.plot(range(1, ROUNDS + 1), best_kpi_values, marker='o')
plt.ylim(min(best_kpi_values) - 0.1, max(best_kpi_values) + 0.1)
plt.xlabel('Optimization Round')
plt.ylabel('Best EFE_m Production (KPI)')
plt.title('Improvement of EFE_m Production Over Optimization Rounds')
plt.grid(True)
plt.show()

print(x)
print(y)
print(f"Best KPI values:", best_kpi_values)