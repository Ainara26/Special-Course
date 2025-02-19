import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process
from skopt.acquisition import gaussian_ei

# Función real (simula los días soleados en Copenhague)
def true_function(x):
    # Sinusoidal para la estacionalidad + ruido aleatorio
    return 40 * np.sin(2 * np.pi * x / 12) + 80 + np.random.normal(0, 5, size=x.shape)

# Función objetivo (sin ruido, para predicción)
def objective_function(x):
    return 40 * np.sin(2 * np.pi * x / 12) + 80

# Rango de meses (1 a 12)
months = np.linspace(1, 12, 100)

# Datos observados iniciales
np.random.seed(42)  # Para reproducibilidad
observed_months = np.array([2, 6, 9])  # Meses en los que ya tenemos datos
observed_sunny_days = true_function(observed_months)

# Visualización inicial de los datos
plt.figure(figsize=(10, 6))
plt.plot(months, objective_function(months), "--", label="Función real (desconocida)")
plt.scatter(observed_months, observed_sunny_days, color="red", label="Datos observados")
plt.title("Días soleados en Copenhague (predicción inicial)")
plt.xlabel("Mes")
plt.ylabel("Días soleados")
plt.legend()
plt.show()

# Optimización Bayesiana usando skopt
def bayesian_optimization(n_iter=5):
    # Convertimos los datos observados a formato 2D
    observed_months_2d = observed_months.reshape(-1, 1)

    # Definir el modelo (proceso gaussiano)
    from skopt import Optimizer
    opt = Optimizer(dimensions=[(1, 12)], base_estimator="gp")

    # Alimentar los datos iniciales al modelo
    for x, y in zip(observed_months_2d, observed_sunny_days):
        opt.tell(x, y)

    # Iteraciones de la optimización bayesiana
    for i in range(n_iter):
        next_point = opt.ask()  # Siguiente punto a evaluar
        next_value = true_function(np.array(next_point))[0]  # Obtener el "valor real" del punto
        opt.tell(next_point, next_value)  # Actualizar el modelo

        # Visualización después de cada iteración
        plt.figure(figsize=(10, 6))
        plot_gaussian_process(opt.space, opt.models[-1], optimizer=opt,
                              n_random_starts=0, show_acq_func=True)
        plt.scatter(observed_months, observed_sunny_days, color="red", label="Datos observados")
        plt.scatter(next_point, next_value, color="blue", label="Nuevo punto evaluado")
        plt.title(f"Optimización Bayesiana: Iteración {i + 1}")
        plt.legend()
        plt.show()

# Ejecutar optimización
bayesian_optimization(n_iter=5)
