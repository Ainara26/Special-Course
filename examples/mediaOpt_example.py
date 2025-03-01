import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.synthetic import Hartmann
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

torch.set_default_dtype(torch.double)

# media components
MEDIA = ["FeSO4", "K2HPO4", "NH4Cl", "CaCl2", "NaCl", "Na2SO4"]

# boundaries
BOUNDS = [(0.0, 1.0)] * len(MEDIA) #minimum (0) and maximum (1) boundaries for each media component (len(MEDIA))
    #BOUNDS = [
    #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Minimum values (lower bound)
    #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # Maximum values (upper bound)
    #]

# shapes
Q = 12 #number of batch of experiments that run in parallel
D = len(MEDIA) #dimensionality of the optimisation problem (in this case, the 6 media components)

#Hartmann function that acts as our simulated objective function
TRUTH = Hartmann(dim=D, bounds=BOUNDS, negate=True, noise_std=0.05)
TRUE_ARGOPTIMUM = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) #best possible combination of media components
TRUE_OPTIMUM = 3.32237 #best results of growth

# Fix random seed to control the randomness of the function
SEED = 12345
torch.manual_seed(seed=SEED)

#We created a function called get_next_batch_of_designs for defining the Bayesian Optimization Process
    #The goal of this function is to select the next set of experimental conditions (designs) that optimize log Expected Improvement (pLogEI).
    #This function generates a batch of new experiments (q points) based on previously tested conditions (x,y).
def get_next_batch_of_designs(
    x: torch.Tensor, #past experimental conditions
    y: torch.Tensor, #observed KPI values, for example growth rate
    bounds: torch.Tensor | list[tuple[float, float]], #min and max value of each variable
    q: int, #number of experiments to generate
    seed: int | None = None, #random seed for reproducibility
) -> tuple[torch.Tensor, float]: #returns (batch of designs, log-scale acquisition value)
    
    #handling random seed
    if seed is not None:
        torch.manual_seed(seed=seed)
    
    #transposes it to match the format required by BoTorch (d x 2)-> each row will have the min and max of each component respectively
    bounds_t = torch.Tensor(bounds).T

    #creation of a GP model that can predict KPI values for untested conditions, along with uncertainty estimates
    #surrogate model
    gp_model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        input_transform=Normalize(d=x.shape[1]),
        outcome_transform=Standardize(m=1),
    )

    prediction=gp_model(x)
    mean=prediction.mean
    print(f"Mean of GP model:",mean.shape)
    print(f"Shape of y:",y.shape)

    #Trains the GP model to accurately approximate the objective function.
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)

    #acquisition function
    sampler = SobolQMCNormalSampler(torch.Size([q]), seed=seed)
    qlei = qLogExpectedImprovement(
        model=gp_model,
        best_f=y.max(),
        sampler=sampler,
    )

    #optimizing the acquisition function
    candidates, joint_acq = optimize_acqf(
        acq_function=qlei,
        bounds=bounds_t,
        q=q,
        num_restarts=2,
        raw_samples=10,
        sequential=False,
    )

    #return new experimental conditions
    return candidates, float(joint_acq)

def main():
    #Converts bounds to a tensor, so each row represents min and max values for a media component.
    bounds_tensor = torch.Tensor(BOUNDS).T
    #Generation of the first batch of experiments, using the sobol sampling (random method)
    first_x = draw_sobol_samples(
        bounds=bounds_tensor,
        q=Q,
        n=1,
        seed=SEED,
    ).squeeze(0)
    rounds = 2 #number of optimization rounds
    x = first_x #store experimental conditions
    y = torch.Tensor([]) #empty tensor to store the KPI data
    best = [] #to keep track of the best KPI values found so far
    pbar = tqdm(range(rounds)) #progress bar for tracking rounds
    next_x = first_x #start with the first batch
    for r in pbar:
        next_y = TRUTH.forward(next_x, noise=True).unsqueeze(-1) #evaluates the KPI values (y) for the current batch (next_x), adding experimental noise
        y = torch.concat([y, next_y]) #updates y with the new KPI values.
            #in each round, we test Q new media compositions and store the KPI values in y--> x and y grows as new experiments are added
        print(y.shape)
        #we call get_next_batch_of_designs--> the optimizer suggests the next batch of Q=12 experiments
        next_x, _ = get_next_batch_of_designs(
            x=x,
            y=y,
            bounds=BOUNDS,
            q=Q,
            seed=SEED,
        )

        best_so_far = float(y.max()) #best KPI so far
        msg = f"Best after round {r}: " + str(best_so_far) #print the message
        pbar.set_description(msg) #update the tracking bar
        best.append([r, best_so_far]) #store the best KPI

        x = torch.concat([x, next_x]) #add new conditions to the dataset
    print(x) #list of all tested media compositions
    plot_kpi_progress(best)

#representation of how KPI variates over the rounds
def plot_kpi_progress(best):
    best = torch.tensor(best)  # Convert to tensor for easy plotting
    rounds = best[:, 0]  # Extract round numbers
    best_kpis = best[:, 1]  # Extract best KPI values

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, best_kpis, marker="o", linestyle="-", color="b", label="Best KPI")
    plt.xlabel("Optimization Round")
    plt.ylabel("Best KPI Found")
    plt.title("Bayesian Optimization Progress: Best KPI Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()