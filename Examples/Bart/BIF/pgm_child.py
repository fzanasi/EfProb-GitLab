from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *

# http://www.bnlearn.com/bnrepository/

reader=BIFReader('child.bif')

model = reader.get_model()

print( model.nodes )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "child")

inference = VariableElimination(model)

#print( inference.query(['Erk'], evidence={'P38': 2}) ['Erk'] )

print( efprob_domains_of_pgm(model) )

channels = efprob_channels_of_pgm(model)

print( channels )

stretch = stretch(model)

#graph_image(stretch['graph'], "sachs")

