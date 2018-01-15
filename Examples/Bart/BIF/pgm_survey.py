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

#stretch = stretch(model)
stretch = stretch(model,graph_output=True,observed=True)

#graph_image(stretch['graph'], "sachs")

print("\n* MAP query")

joint = evaluate_stretch(stretch['channels'])
print( joint.MAP() )

print( inference.map_query(variables=model.nodes) )



#print( inference_query(stretch, 'O', {'S' : [0,1]}) )


