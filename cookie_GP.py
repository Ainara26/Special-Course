import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid of temperature and time
temperature = np.linspace(150, 200, 30)  # Temperature range (150-200°C)
time = np.linspace(8, 15, 30)  # Time range (8-15 minutes)
temperature_grid, time_grid = np.meshgrid(temperature, time)

# Define the "true" cookie quality function (unknown to the optimizer)
def cookie_quality(temp, time):
    return np.exp(-((temp - 180)**2 / 50 + (time - 12)**2 / 4)) * 9

# Generate the true cookie quality (used for understanding, not for the optimizer)
true_quality = cookie_quality(temperature_grid, time_grid)

# Define the surrogate model's initial mean predictions
surrogate_mean = np.exp(-((temperature_grid - 180)**2 / 100 + (time_grid - 12)**2 / 8)) * 6

# Define the uncertainty of the surrogate model
uncertainty = np.ones_like(surrogate_mean) * 2  # Constant uncertainty initially

# Plot the surrogate model mean and uncertainty as surfaces
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the surrogate mean
surf = ax.plot_surface(temperature_grid, time_grid, surrogate_mean, cmap="viridis", alpha=0.8, edgecolor="none")
ax.set_title("Bayesian Optimization: Surrogate Model for Cookie Quality", fontsize=14)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Time (min)")
ax.set_zlabel("Predicted Quality (Surrogate Mean)")

# Overlay uncertainty as transparent bounds
ax.plot_surface(temperature_grid, time_grid, surrogate_mean + uncertainty, color="red", alpha=0.2, edgecolor="none")
ax.plot_surface(temperature_grid, time_grid, surrogate_mean - uncertainty, color="blue", alpha=0.2, edgecolor="none")

# Add observed data points (example values)
observed_temps = np.array([160, 180, 170])  # Observed temperatures
observed_times = np.array([10, 12, 9])  # Observed times
observed_scores = np.array([6, 8, 5])  # Observed quality scores
ax.scatter(observed_temps, observed_times, observed_scores, color="black", s=50, label="Observed Data", edgecolors="white")

# Add a legend
ax.legend(["Surrogate Mean", "Uncertainty Bounds", "Observed Data"], loc="upper left")

# Show the plot
plt.show()
