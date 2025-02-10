from cobra.io import read_sbml_model

model = read_sbml_model('C:/Users/Ainara/Documents/GitHub/Special-Course/models/modified_model.xml')

#Define the Search Space
MEDIA=model.medium
print (MEDIA.keys)