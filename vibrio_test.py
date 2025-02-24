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
    MEDIA = model.medium
    MEDIA['EX_Na+_e'] = 20.0
    MEDIA['EX_Acetate_e']=2.5
    MEDIA['EX_NH3_e']=1.0
    MEDIA['EX_K+_e']=0.2
    MEDIA['EX_Mg_e']=0.2
    for exchange in model.medium.keys():
        model.medium[exchange] = 1000.0
    print(f"Max theoretical growth:", model.optimize().objective_value)