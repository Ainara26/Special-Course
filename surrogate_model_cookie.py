import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define the search space
temp = np.linspace(150, 250, 50)  # Temperature range
time = np.linspace(5, 15, 50)    # Time range
temp_grid, time_grid = np.meshgrid(temp, time)

# Step 2: Define a surrogate model (Gaussian Process-like)
# True function (just for demonstration)
def true_function(temp, time):
    return np.exp(-0.01 * ((temp - 200) ** 2 + (time - 10) ** 2)) * 10

# Surrogate mean prediction (initial, based on limited data)
mean_prediction = np.exp(-0.01 * ((temp_grid - 200) ** 2 + (time_grid - 10) ** 2)) * 10

# Uncertainty (arbitrarily larger initially, decreases near sampled points)
uncertainty = 5 - 0.01 * ((temp_grid - 200) ** 2 + (time_grid - 10) ** 2)

# Step 3: Sampled points
sampled_points = [(200, 10), (180, 8), (220, 12)]  # Example points
sampled_values = [true_function(x[0], x[1]) for x in sampled_points]

# Step 4: Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the mean prediction
ax.plot_surface(temp_grid, time_grid, mean_prediction, cmap='viridis', alpha=0.8, label='Mean Prediction')

# Plot the uncertainty as a wireframe
ax.plot_wireframe(temp_grid, time_grid, mean_prediction + uncertainty, color='red', alpha=0.3, label='Upper Confidence')
ax.plot_wireframe(temp_grid, time_grid, mean_prediction - uncertainty, color='blue', alpha=0.3, label='Lower Confidence')

# Add sampled points
for i, (temp_point, time_point) in enumerate(sampled_points):
    ax.scatter(temp_point, time_point, sampled_values[i], color='black', s=50, label='Sampled Point' if i == 0 else None)

# Labels and title
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Time (minutes)')
ax.set_zlabel('Cookie Quality')
ax.set_title('Bayesian Optimization: Mean and Uncertainty')

plt.legend(loc='upper right')
plt.show()
