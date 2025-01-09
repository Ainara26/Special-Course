from cobra.io import read_sbml_model
model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iJN1463.xml')
# Specific reaction names/keywords for ED pathway
ed_keywords = ["phosphogluconate", "KDPG", "glucose-6-phosphate"]

# Filter reactions related to ED pathway
for reaction in model.reactions:
    if any(keyword in reaction.name.lower() for keyword in ed_keywords):
        print(f"Reaction: {reaction.id} | Name: {reaction.name} | GPR: {reaction.gene_reaction_rule}")
