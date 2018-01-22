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

#model.remove_node('ChestXray')

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "child")

picks = pick_from_list(model.nodes, 5)

inference = VariableElimination(model)

evidence_dictionary = {}
for e in picks[2:]:
    ls = model.get_cardinality(e) * [0]
    ls[0] = 1
    evidence_dictionary[e] = ls


print("\nInference")
print("=========")

print("\n* Via transformations")
print( stretch_and_infer(model, picks[0], evidence_dictionary) )

print("\n* Via variable elimination")
print( inference.query([picks[0]], evidence={picks[2]: 0, picks[3]: 0, picks[4] : 0})
       [picks[0]] )



N = 10

t1 = timeit.timeit(lambda: inference.query([picks[0]], 
                                           evidence={picks[2]: 0, 
                                                     picks[3]: 0, 
                                                     picks[4] : 0})[picks[0]], 
                   number = N)

t2 = timeit.timeit(lambda: stretch_and_infer(model, picks[0], 
                                             evidence_dictionary), 
                   number = N)

print("\nTimes for: variable elimination, transformations, fraction, for", 
N, "runs")
print(t1)
print(t2)
print("How much faster is transformations inference:", t1/t2)



print("\nMAP query")
print("=========")

print("\n* Via transformations")
print( inference_map_query(model, variables=[picks[0], picks[1]],
                           evidence_dict = evidence_dictionary, silent=True) )

print("\n* Via map-query")
print( inference.map_query(variables=[picks[0], picks[1]],
                           evidence={picks[2]: 0, picks[3]: 0, picks[4] : 0}) )




"""

stretch = stretch(model,graph_output=True)

#graph_image(stretch['graph'], "child")

N = 10


print( inference.query(['LowerBodyO2'], evidence={'Age': 2, 'LungFlow' : 1})
       ['LowerBodyO2'] )

# t1 = timeit.timeit(lambda: 
#                    inference.query(['LowerBodyO2'], 
#                                    evidence={'Age': 2, 'LungFlow' : 1})
#                    ['LowerBodyO2'],
#                    number = N)

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


"""
