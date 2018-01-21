from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *

# http://www.bnlearn.com/bnrepository/

reader=BIFReader('survey.bif')

model = reader.get_model()

#print( model.nodes )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "survey")

inference = VariableElimination(model)

#print( inference.query(['O'], evidence={'S': 1}) ['O'] )

#print( efprob_domains_of_pgm(model) )

#channels = efprob_channels_of_pgm(model)

#print( channels )

#survey_stretch = stretch(model)
survey_stretch = stretch(model,graph_output=True,observed=True)

#graph_image(survey_stretch['graph'], "sachs")

print("\n* MAP query")

joint = evaluate_stretch(survey_stretch['channels'])
print( joint.dom.names )
print( (joint % [0,0,1,1,0,0]).MAP() )

vars = ['E', 'R']

print( inference.map_query(variables=vars) )

print("")

survey_stretch = stretch(model)

print( inference_map_query(survey_stretch, variables = vars) )



#print( inference_query(stretch, 'O', {'S' : [0,1]}) )


