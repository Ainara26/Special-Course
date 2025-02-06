import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch 
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.synthetic import Ackley
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm


#to select the test function based on the dimensionality
def get_test_function(D, bounds):
    if D==6:
        print("Using Hartmann Function for Optimization")
        return Hartmann(dim=6, bounds=bounds, negate=True, noise_std=0.05)
    if D>6 and D<=25:
        print("Using Ackley Function for Optimization")
        return Ackley(dim=D, bounds=bounds, negate=True, noise_std=0.05)
    else:
        print("No synthetic test function used (Expecting real-world data)")
        return None

#train of surrogate model
def train_gp_model(x, y, D):
    """
    Trains a Gaussian Process model.
    - Uses SingleTaskGP for D ≤ 25.
    - Uses SaasGP (faster) for D > 25.
    """
    # ✅ Ensure `y` has shape (N, 1)
    if y.dim() == 1:
        y = y.unsqueeze(-1)

    if D <= 25:
        gp_model = SingleTaskGP(train_X=x, train_Y=y)  # ✅ DO NOT add likelihood here
    else:
        gp_model = SaasGP(train_X=x, train_Y=y)  # ✅ DO NOT add likelihood here

    # ✅ Remove likelihood from `mll`
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)  
    fit_gpytorch_mll(mll)
    
    return gp_model

#to obtain the next set of experiments
def get_next_batch_of_designs(
    x: torch.Tensor, #past experimental conditions
    y: torch.Tensor, #observed KPI values, for example growth rate
    bounds: torch.Tensor | list[tuple[float, float]], #min and max value of each variable
    q: int, #number of experiments to generate
    seed: int | None = None, #random seed for reproducibility
) -> tuple[torch.Tensor, float]: #returns (batch of designs, log-scale acquisition value)
    
    if seed is not None:
        torch.manual_seed(seed=seed)
    
    #transposes it to match the format required by BoTorch (d x 2)-> each row will have the min and max of each component respectively
    bounds_t = torch.Tensor(bounds).T
    D = x.shape[1]
    gp_model = train_gp_model(x, y, D)

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

#bayesian loop
def bayesian_optimization(x, y, bounds, Q, rounds, seed, test_function):
    torch.manual_seed(seed)
    best_kpis = []

    for r in range(rounds):
        print(f"\n--- ROUND {r+1}/{rounds} ---")
        
        # Get the next batch of experiments
        next_x, _ = get_next_batch_of_designs(x, y, bounds, Q, seed)

        # Evaluate KPI using test function (if available)
        if test_function:
            next_y = test_function.forward(next_x, noise=True).unsqueeze(-1)
        else:
            print("Real-world KPI values needed for next_y!")
            return x, y, best_kpis

        # Update dataset
        x = torch.cat([x, next_x])
        y = torch.cat([y, next_y])

        # Track best KPI
        best_kpi = float(y.max())
        best_kpis.append(best_kpi)
        print(f"Best KPI so far: {best_kpi:.3f}")

    return x, y, best_kpis

#visualization of how KPI improves over iterations
def plot_kpi_progress(best_kpis):
    rounds = range(1, len(best_kpis) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, best_kpis, marker="o", linestyle="-", color="b", label="Best KPI")
    plt.xlabel("Optimization Round")
    plt.ylabel("Best KPI Found")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


