from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *
import timeit

# ask for swappiness:
#   sysctl vm.swappiness
# normally it as at 60; set it to 0 to prevent freezing, via:
#   sudo sysctl -w vm.swappiness=0

# http://www.bnlearn.com/bnrepository/discrete-small.html#sachs

print("\nSachs Bayesian network")
print("======================\n")

reader=BIFReader('sachs.bif')

model = reader.get_model()

#print( model.nodes )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "sachs")

print("\nStretching the graph:")

#stretch = stretch(model,graph_output=True)
stretch = stretch(model,graph_output=False)
#stretch = stretch(model,graph_output=True,observed=True)

#graph_image(stretch['graph'], "sachs")

print("\nVariable elimination inference")

N = 1

inference = VariableElimination(model)

print( inference.query(['Erk'], 
                       evidence={'P38' : 2, 'Jnk' : 1})['Erk'] )

t1 = timeit.timeit(lambda: 
                   inference.query(['Erk'], 
                                   evidence={'P38': 2, 'Jnk' : 1}) ['Erk'],
                   number = N)

print("\nTransformations inference")

#stretch = stretch(model)

print( inference_query(stretch, 'Erk', {'P38' : [0,0,1], 'Jnk' : [0,1,0]}) )

t2 = timeit.timeit(lambda: 
                   inference_query(stretch, 'Erk', 
                                   {'P38' : [0,0,1], 'Jnk' : [0,1,0]}),
                   number = N)

print("\nTimes for: variable elimination, transformations, fraction, for", 
N, "runs")
print(t1)
print(t2)
print("How much beter is transformations inference:", t1/t2)


