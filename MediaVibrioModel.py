import cobra
import torch
from cobra.io import read_sbml_model

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iLC858.sbml')
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

solution = model.optimize()
print(f"Max growth for baseline media:", solution)