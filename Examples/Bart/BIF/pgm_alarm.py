from pgm_efprob import *
# from pgmpy.models import BayesianModel
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import *
import timeit

# http://www.bnlearn.com/bnrepository/

reader=BIFReader('alarm.bif')

model = reader.get_model()

#print( efprob_domains_of_pgm(model) )

graph = pydot_graph_of_pgm(model)

#graph_image(graph, "alarm")

picks = pick_from_list(model.nodes, 3)

evidence_dictionary = {}
for e in picks[1:]:
    ls = model.get_cardinality(e) * [0]
    ls[0] = 1
    evidence_dictionary[e] = ls

print( picks )

inference = VariableElimination(model)

print( inference.query([picks[0]], evidence={picks[1]: 0, picks[2] : 0})
       [picks[0]] )

print( stretch_and_infer(model, picks[0], evidence_dictionary, silent=True) )

#print( stretch_and_infer(model, 'FIO2', ['ANAPHYLAXIS', 'VENTTUBE'], silent=False) )


N = 5

t1 = timeit.timeit(lambda: inference.query([picks[0]], 
                                           evidence={picks[1]: 0, 
                                                     picks[2] : 0})[picks[0]], 
                   number = N)

print(t1)

t2 = timeit.timeit(lambda: stretch_and_infer(model, picks[0], 
                                             evidence_dictionary), 
                   number = N)

print("\nTimes for: variable elimination, transformations, fraction, for", 
N, "runs")
print(t2)
print("How much beter is transformations inference:", t1/t2)



