import cobra
import torch
from cobra.io import read_sbml_model
import itertools 

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iLC858.sbml')
# Get all exchange reactions available in the model
exchange_reactions = [rxn.id for rxn in model.exchanges]

# Initialize a new medium with only the desired components (if they exist)
new_medium = {}
# Define desired nutrients and their concentrations
custom_medium = {
    'EX_cpd00099_e': 1000.0, #clhoride
    'EX_cpd00971_e': 1000.0, #sodium
    'EX_cpd00048_e': 1000.0, #sulfate
    'EX_cpd00205_e': 1000.0, #potassium
    'EX_cpd00063_e': 1000.0, #calcium
    'EX_cpd00149_e': 1000.0, #carbonate
    'EX_cpd00254_e': 1000.0, #magnesium
    'EX_cpd00029_e': 1000.0, #acetate
    'EX_cpd00013_e': 1000.0, #ammonium
    'EX_cpd00012_e': 1000.0, #H2PO4
    'EX_cpd00305_e': 1000.0, #thiamine/vit B1
    'EX_cpd00635_e': 1000.0, #vit B12
    'EX_cpd00028_e': 1000.0  #iron
}

# Only add nutrients that exist as exchange reactions in the model
for nutrient, concentration in custom_medium.items():
    if nutrient in exchange_reactions:
        new_medium[nutrient] = concentration
    else:
        print(f"Warning: {nutrient} not found in model exchange reactions.")

# Update the model's medium
model.medium = new_medium
print(model.medium, f"Length custom medium:", len(custom_medium))

solution = model.optimize()
print(f"Max growth for baseline media:", solution)

essential_components = {
    'EX_cpd00009_e': 1000.0,  # Phosphate
    'EX_cpd00011_e': 1000.0,  # CO2
    'EX_cpd00012_e': 1000.0,  # PPi
    'EX_cpd00013_e': 1000.0,  # NH3
    'EX_cpd00030_e': 1000.0,  # Mn2+
    'EX_cpd00034_e': 1000.0,  # Zn2+
    'EX_cpd00048_e': 1000.0,  # Sulfate
    'EX_cpd00058_e': 1000.0,  # Cu2+
    'EX_cpd00063_e': 1000.0,  # Ca2+
    'EX_cpd00067_e': 1000.0,  # H+
    'EX_cpd00099_e': 1000.0,  # Cl-
    'EX_cpd00149_e': 1000.0,  # Co2+
    'EX_cpd00205_e': 1000.0,  # K+
    'EX_cpd00254_e': 1000.0,  # Mg2+
    'EX_cpd00531_e': 1000.0,  # Hg2+
    'EX_cpd00971_e': 1000.0,  # Na+
    'EX_cpd04097_e': 1000.0,  # Pb
    'EX_cpd10516_e': 1000.0   # Fe+3
}

# Start with your custom medium
base_medium = custom_medium.copy()

for component, concentration in essential_components.items():
    print(f"Testing {component}:")

    # Add one extra component on top of your custom medium
    test_medium = base_medium.copy()
    test_medium[component] = concentration
    
    # Apply the new medium
    model.medium = test_medium
    
    solution = model.optimize()
    print(f"{component} --> Growth: {solution.objective_value:.6f}")
    print(f"length test medium:", len(test_medium))

    # Stop the loop if growth is detected
    if solution.objective_value > 0:
        print(f"BINGO! {component} is the missing piece!")
        break

triplets = list(itertools.combinations(essential_components, 3))

for triplet in triplets:
    medium = {c: 1000.0 for c in triplet}
    model.medium = medium
    solution = model.optimize()
    print(f"{triplet} → Growth: {solution.objective_value}")
