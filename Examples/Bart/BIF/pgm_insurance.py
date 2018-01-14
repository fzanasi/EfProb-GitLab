from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *
import timeit

# http://www.bnlearn.com/bnrepository/

reader=BIFReader('insurance.bif')

model = reader.get_model()

#print( efprob_domains_of_pgm(model) )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "insurance")

#stretch = stretch(model,graph_output=True)

#graph_image(stretch['graph'], "insurance")

picks = pick_from_list(model.nodes, 3)

evidence_dictionary = {}
for e in picks[1:]:
    ls = model.get_cardinality(e) * [0]
    ls[0] = 1
    evidence_dictionary[e] = ls

inference = VariableElimination(model)

print( stretch_and_infer(model, picks[0], evidence_dictionary) )

print( inference.query([picks[0]], evidence={picks[1]: 0, picks[2] : 0})
       [picks[0]] )

N = 5

t1 = timeit.timeit(lambda: inference.query([picks[0]], 
                                           evidence={picks[1]: 0, 
                                                     picks[2] : 0})[picks[0]], 
                   number = N)

t2 = timeit.timeit(lambda: stretch_and_infer(model, picks[0], 
                                             evidence_dictionary), 
                   number = N)

print("\nTimes for: variable elimination, transformations, fraction, for", 
N, "runs")
print(t1)
print(t2)
print("How much beter is transformations inference:", t1/t2)



# t1 = timeit.timeit(lambda: 
#                    inference.query(['Theft'], evidence={'GoodStudent': 0})['Theft'],
#                    number = N)



"""


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
