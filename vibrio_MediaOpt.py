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
# Get all exchange reactions available in the model
exchange_reactions = [rxn.id for rxn in model.exchanges]

# Initialize a new medium with only the desired components (if they exist)
new_medium = {}

# Define desired nutrients and their concentrations
custom_medium = {
    'EX_cpd00099_e': 64.8, #clhoride
    'EX_cpd00971_e': 91.6, #sodium
    'EX_cpd00048_e': 121.0, #sulfate
    'EX_cpd00205_e': 136.3, #potassium
    'EX_cpd00063_e': 0.2, #calcium
    'EX_cpd00149_e': 0.1, #carbonate
    'EX_cpd00254_e': 121.6, #magnesium
    'EX_cpd00029_e': 82.0, #acetate
    'EX_cpd00013_e': 53.5, #ammonium
    'EX_cpd00012_e': 136.1, #H2PO4
    'EX_cpd00305_e': 0.014, #thiamine/vit B1
    'EX_cpd00635_e': 0.0007378, #vit B12
    'EX_cpd00028_e': 0.010790591 #iron
}

# Only add nutrients that exist as exchange reactions in the model
for nutrient, concentration in custom_medium.items():
    if nutrient in exchange_reactions:
        new_medium[nutrient] = concentration
    else:
        print(f"Warning: {nutrient} not found in model exchange reactions.")

# Update the model's medium
model.medium = new_medium

# Define bounds dynamically based on the new medium
BOUNDS = [(0.001, 10 * value) for value in new_medium.values()]
Q=12
D=len(new_medium)
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