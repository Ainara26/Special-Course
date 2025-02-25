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

# Set all exchange reactions to high values to test max growth
with model:
    for exchange in model.medium.keys():
        model.medium[exchange] = 1000.0
    print(f"Max theoretical growth:", model.optimize().objective_value)

solution = model.optimize()
shadow_prices = solution.reduced_costs
shadow_prices = solution.reduced_costs

# Create a list of limiting nutrients (negative shadow prices)
limiting_nutrients = []
for rxn_id, price in shadow_prices.items():
    if "EX_" in rxn_id and price < 0:  # Only consider exchange reactions with negative values
        reaction = model.reactions.get_by_id(rxn_id)
        metabolite = list(reaction.metabolites.keys())[0]  # Get the associated metabolite
        limiting_nutrients.append((metabolite.name, price))

# Sort by shadow price (most negative first)
limiting_nutrients.sort(key=lambda x: x[1])  # Sort by price (ascending, more negative first)

# Print the ranked list of limiting nutrients
print("**Ranked Limiting Nutrients**:")
for rank, (metabolite_name, price) in enumerate(limiting_nutrients, start=1):
    print(f"{rank}. {metabolite_name}: {price:.4f}")