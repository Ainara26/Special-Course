import cobra
import torch
from cobra.io import read_sbml_model

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iLC858.sbml')
media=model.medium
# Map exchange reactions to human-readable metabolite names
for reaction_id, flux in media.items():
    reaction = model.reactions.get_by_id(reaction_id)
    metabolite = list(reaction.metabolites.keys())[0]  # Get the first metabolite in the reaction
    if "_e" in metabolite.id: 
        print(f"{reaction_id}: {metabolite.name} ({flux} mmol/gDW/h)")

# Set all exchange reactions to high values to test unconstrained growth
for exchange in model.medium.keys():
    model.medium[exchange] = 1000.0

print(f"Max theoretical growth:", model.optimize().objective_value)