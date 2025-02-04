#FBA MEDIUM OPT
import cobra
from cobra.io import read_sbml_model
model=read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/iJO1366.xml')

#Model Objective set as the max growth
model.objective="BIOMASS_Ec_iJO1366_core_53p95M"

def simulate_fba(media_composition):
    with model:
        # Set uptake rates (bounds on exchange reactions)
        model.reactions.get_by_id("EX_glc__D_e").lower_bound = -media_composition[0]  # Glucose uptake
        model.reactions.get_by_id("EX_o2_e").lower_bound = -media_composition[1]  # Oxygen uptake
        model.reactions.get_by_id("EX_nh4_e").lower_bound = -media_composition[2]  # Ammonium uptake
        
        # Run FBA optimization
        solution = model.optimize()
        return solution.objective_value  # Growth rate
