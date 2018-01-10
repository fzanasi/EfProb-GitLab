from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *
import timeit

# http://www.bnlearn.com/bnrepository/

reader=BIFReader('child.bif')

model = reader.get_model()

#print( efprob_domains_of_pgm(model) )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "child")

stretch = stretch(model,graph_output=True)

#graph_image(stretch['graph'], "child")

N = 10

inference = VariableElimination(model)

print( inference.query(['LowerBodyO2'], evidence={'Age': 2, 'LungFlow' : 1})
       ['LowerBodyO2'] )

t1 = timeit.timeit(lambda: 
                   inference.query(['LowerBodyO2'], 
                                   evidence={'Age': 2, 'LungFlow' : 1})
                   ['LowerBodyO2'],
                   number = N)

print("\nTransformations inference")

print( inference_query(stretch, 'LowerBodyO2', 
                       {'Age' : [0,0,1], 'LungFlow' : [0,1,0]}) )

t2 = timeit.timeit(lambda: 
                   inference_query(stretch, 'LowerBodyO2', 
                   {'Age' : [0,0,1], 'LungFlow' : [0,1,0]}),
                   number = N)

print("\nTimes for: variable elimination, transformations, fraction, for", 
N, "runs")
print(t1)
print(t2)
print("How much beter is transformations inference:", t1/t2)


