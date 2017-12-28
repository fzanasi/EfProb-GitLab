from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *

# sudo sysctl -w vm.swappiness=5

# http://www.bnlearn.com/bnrepository/discrete-small.html#sachs

# Memory problems:
# https://askubuntu.com/questions/41778/computer-freezing-on-almost-full-ram-possibly-disk-cache-problem

reader=BIFReader('sachs.bif')

model = reader.get_model()

#print( model.nodes )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "sachs")

inference = VariableElimination(model)

#print( inference.query(['Erk'], evidence={'P38': 2}) ['Erk'] )

print( efprob_domains_of_pgm(model) )

# P38_dom = efprob_domain('P38', 3)

# print( P38_dom )

# p = point_pred('P38_2', P38_dom)

channels = efprob_channels_of_pgm(model)

#print( channels )

stretch = stretch(model,graph_output=False)

#graph_image(stretch['graph'], "sachs")

