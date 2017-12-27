#
# model class code at:
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/models/BayesianModel.py
#
# TabularCPD in:
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/CPD.py
#
from efprob_dc import *
from baynet import *

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#from pgmpy.inference import BeliefPropagation

import pydot
from PIL import Image

"""



print("\nStudent model")
print("=============\n")

# copied from:
# https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb

student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# Defining individual CPDs.
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])

cpd_g = TabularCPD(variable='G', variable_card=3, 
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                   evidence=['I', 'D'],
                   evidence_card=[2, 2])

cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])

student_model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

print("\nModel correct: ",  student_model.check_model() )

"""


#print("\nChannels\n")

#print( efprob_channel_from_tabularCPD(cpd_ashtray) )
#print( efprob_channel_from_tabularCPD(cpd_g) )
#print( efprob_channel_from_tabularCPD(cpd_l) )


#print("\nStudent joint")


#student_graph = pydot_graph_of_pgm(student_model)
#graph_image(student_graph, "student")

#student_cpts = efprob_channels_of_pgm(student_model)
#student_domain = reduce(lambda d1, d2: d1 + d2, 
#                        efprob_domains_of_pgm(student_model))

#student_joint = reorder_state_domains(flatten(student_graph, student_cpts), 
#                                      ['D', 'I', 'G', 'L', 'S'])

#print( student_joint )

#print("\nMarginals")
#print( student_joint % [1,0,0,0,0] )
#print( student_inference.query(['D'])['D'] )
#print( student_joint % [0,1,0,0,0] )
#print( student_inference.query(['I'])['I'] )
#print( student_joint % [0,0,1,0,0] )
#print( student_cpts['G'] >> (student_cpts['I'] @ student_cpts['D']) )
#print( student_inference.query(['G'])['G'] )
#print( student_joint % [0,0,0,0,1] )
#print( student_inference.query(['S'])['S'] )
#print( student_joint % [0,0,0,1,0] )
#print( student_inference.query(['L'])['L'] )

# print( student_cpts['G'] )
# print( student_cpts['G']('I_0', 'D_0') )
# print( student_cpts['G']('I_0', 'D_1') )
# print( student_cpts['G']('I_1', 'D_0') )
# print( student_cpts['G']('I_1', 'D_1') )

#print( cpd_g )



#print("\nUpdate test")

#student_dom = student_joint.dom

#p = point_pred('S_0', student_dom[4])

#print( student_joint / (truth(student_dom[0:4]) @ p) % [0,0,0,1,0] )

#print( student_cpts['L'] >> (student_cpts['G'] >> ((student_cpts['I']/(student_cpts['S'] << p)) @ student_cpts['D'])) )

#student_inference = VariableElimination(student_model)

#print( student_inference.query(['L'], evidence={'S': 0})['L'] )


import timeit

N = 100
    
# print(timeit.timeit(lambda: student_joint/(truth(student_dom[0:4]) @ p) % [0,0,0,1,0], 

# #print(timeit.timeit(lambda: flatten(pydot_graph_of_pgm(student_model), 
# #                                    efprob_channels_of_pgm(student_model)), 
# #                    number=N))

# print(timeit.timeit(lambda: student_inference.query(['L'], 
#                                                     evidence={'S': 0})['L'], 
#                     number=N))


#print("\nVisit to Asia")

# https://github.com/pgmpy/pgmpy_notebook/blob/72e2eb65777ddde9bc48554de2b07290f94d1d3d/notebooks/8.%20Reading%20and%20Writing%20from%20pgmpy%20file%20formats.ipynb

edges_list = [('VisitToAsia', 'Tuberculosis'),
              ('LungCancer', 'TuberculosisOrCancer'),
              ('Smoker', 'LungCancer'),
              ('Smoker', 'Bronchitis'),
              ('Tuberculosis', 'TuberculosisOrCancer'),
              ('Bronchitis', 'Dyspnea'),
              ('TuberculosisOrCancer', 'Dyspnea'),
              ('TuberculosisOrCancer', 'Xray')]
nodes = {'Smoker': {'States': {'no': {}, 'yes': {}},
                    'role': 'chance',
                    'type': 'finiteStates',
                    'Coordinates': {'y': '52', 'x': '568'},
                    'AdditionalProperties': {'Title': 'S', 'Relevance': '7.0'}},
         'Bronchitis': {'States': {'no': {}, 'yes': {}},
                        'role': 'chance',
                        'type': 'finiteStates',
                        'Coordinates': {'y': '181', 'x': '698'},
                        'AdditionalProperties': {'Title': 'B', 'Relevance': '7.0'}},
         'VisitToAsia': {'States': {'no': {}, 'yes': {}},
                         'role': 'chance',
                         'type': 'finiteStates',
                         'Coordinates': {'y': '58', 'x': '290'},
                         'AdditionalProperties': {'Title': 'A', 'Relevance': '7.0'}},
         'Tuberculosis': {'States': {'no': {}, 'yes': {}},
                          'role': 'chance',
                          'type': 'finiteStates',
                          'Coordinates': {'y': '150', 'x': '201'},
                          'AdditionalProperties': {'Title': 'T', 'Relevance': '7.0'}},
         'Xray': {'States': {'no': {}, 'yes': {}},
                   'role': 'chance',
                   'AdditionalProperties': {'Title': 'X', 'Relevance': '7.0'},
                   'Coordinates': {'y': '322', 'x': '252'},
                   'Comment': 'Indica si el test de rayos X ha sido positivo',
                   'type': 'finiteStates'},
         'Dyspnea': {'States': {'no': {}, 'yes': {}},
                     'role': 'chance',
                     'type': 'finiteStates',
                     'Coordinates': {'y': '321', 'x': '533'},
                     'AdditionalProperties': {'Title': 'D', 'Relevance': '7.0'}},
         'TuberculosisOrCancer': {'States': {'no': {}, 'yes': {}},
                                  'role': 'chance',
                                  'type': 'finiteStates',
                                  'Coordinates': {'y': '238', 'x': '336'},
                                  'AdditionalProperties': {'Title': 'E', 'Relevance': '7.0'}},
         'LungCancer': {'States': {'no': {}, 'yes': {}},
                        'role': 'chance',
                        'type': 'finiteStates',
                        'Coordinates': {'y': '152', 'x': '421'},
                        'AdditionalProperties': {'Title': 'L', 'Relevance': '7.0'}}}
edges = {'LungCancer': {'TuberculosisOrCancer': {'directed': 'true'}},
         'Smoker': {'LungCancer': {'directed': 'true'},
                    'Bronchitis': {'directed': 'true'}},
         'Dyspnea': {},
         'Xray': {},
         'VisitToAsia': {'Tuberculosis': {'directed': 'true'}},
         'TuberculosisOrCancer': {'Xray': {'directed': 'true'},
                                  'Dyspnea': {'directed': 'true'}},
         'Bronchitis': {'Dyspnea': {'directed': 'true'}},
         'Tuberculosis': {'TuberculosisOrCancer': {'directed': 'true'}}}

cpds = [{'Values': np.array([[0.95, 0.05], [0.02, 0.98]]),
         'Variables': {'Xray': ['TuberculosisOrCancer']}},
        {'Values': np.array([[0.7, 0.3], [0.4,  0.6]]),
         'Variables': {'Bronchitis': ['Smoker']}},
        {'Values':  np.array([[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]),
         'Variables': {'Dyspnea': ['TuberculosisOrCancer', 'Bronchitis']}},
        {'Values': np.array([[0.9], [0.01]]),
        #{'Values': np.array([[0.99], [0.01]]),
         'Variables': {'VisitToAsia': []}},
        {'Values': np.array([[0.5], [0.5]]),
         'Variables': {'Smoker': []}},
        {'Values': np.array([[0.99, 0.01], [0.9, 0.1]]),
         'Variables': {'LungCancer': ['Smoker']}},
        {'Values': np.array([[0.99, 0.01], [0.95, 0.05]]),
         'Variables': {'Tuberculosis': ['VisitToAsia']}},
        {'Values': np.array([[1, 0, 0, 1], [0, 1, 0, 1]]),
         'Variables': {'TuberculosisOrCancer': ['LungCancer', 'Tuberculosis']}}]


asia_model = BayesianModel(edges_list)

# for node in nodes:
#     model.node[node] = nodes[node]
# for edge in edges:
#     model.edge[edge] = edges[edge]

# tabular_cpds = []
# for cpd in cpds:
#     var = list(cpd['Variables'].keys())[0]
#     evidence = cpd['Variables'][var]
#     values = cpd['Values']
#     states = len(nodes[var]['States'])
#     evidence_card = [len(nodes[evidence_var]['States'])
#                      for evidence_var in evidence]
#     tabular_cpds.append(
#         TabularCPD(var, states, values, evidence, evidence_card))

# asia_model.add_cpds(*tabular_cpds)

cpd_smoker = TabularCPD(variable='Smoker', variable_card=2,
                        values=[[0.5], [0.5]])

cpd_visittoasia = TabularCPD(variable='VisitToAsia', variable_card=2,
                             values=[[0.01], [0.99]])

cpd_lungcancer = TabularCPD(variable='LungCancer', variable_card=2, 
                            values=[[0.1, 0.01], [0.9, 0.99]],
                            evidence=['Smoker'], evidence_card=[2])

cpd_tuberculosis = TabularCPD(variable='Tuberculosis', variable_card=2, 
                              values=[[0.05, 0.01], [0.95, 0.99]],
                              evidence=['VisitToAsia'], evidence_card=[2])

cpd_bronchitis = TabularCPD(variable='Bronchitis', variable_card=2, 
                            values=[[0.6, 0.3], [0.4, 0.7]],
                            evidence=['Smoker'], evidence_card=[2])

cpd_tuberculosisorcancer = TabularCPD(variable='TuberculosisOrCancer', variable_card=2, 
                    # [[1, 0, 0, 1], [0, 1, 0, 1]]
                   values=[[1, 0, 0, 0],
                           [0, 1, 1, 1]],
                   evidence=['LungCancer', 'Tuberculosis'],
                   evidence_card=[2, 2])

cpd_dyspnea = TabularCPD(variable='Dyspnea', variable_card=2, 
                    # [[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]
                   values=[[0.9, 0.7, 0.8, 0.1],
                           [0.1, 0.3, 0.2, 0.9]],
                   evidence=['Bronchitis', 'TuberculosisOrCancer'],
                   evidence_card=[2, 2])

cpd_xray = TabularCPD(variable='Xray', variable_card=2, 
                      values=[[0.98, 0.05], [0.02, 0.95]],
                      evidence=['TuberculosisOrCancer'], evidence_card=[2])

asia_model.add_cpds(cpd_smoker, cpd_visittoasia, 
                    cpd_lungcancer, cpd_tuberculosis,
                    cpd_bronchitis, cpd_tuberculosisorcancer,
                    cpd_dyspnea, cpd_xray)


#print("\nAsia correct: ", asia_model.check_model() )

asia_inference = VariableElimination(asia_model)

asia_graph = pydot_graph_of_pgm(asia_model)
#graph_image(asia_graph, "asia")

asia_cpts = efprob_channels_of_pgm(asia_model)

# print( asia_cpts['Smoker'] )
# print( asia_cpts['VisitToAsia'] )
# print( asia_cpts['LungCancer']('Smoker_0') )
# print( asia_cpts['LungCancer']('Smoker_1') )
# #print( asia_cpts['Tuberculosis'] )
# print( asia_cpts['Tuberculosis']('VisitToAsia_0') )
# print( asia_cpts['Tuberculosis']('VisitToAsia_1') )
# #print( asia_cpts['Bronchitis'] )
# print( asia_cpts['Bronchitis']('Smoker_0') )
# print( asia_cpts['Bronchitis']('Smoker_1') )
# #print( asia_cpts['TuberculosisOrCancer'] )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_0') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_1') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_0') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_1') )
# #print( asia_cpts['Dyspnea'] )
# print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_0') )
# print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_1') )
# print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_0') )
# print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_1') )
# #print( asia_cpts['Xray'])
# print( asia_cpts['Xray']('TuberculosisOrCancer_0') )
# print( asia_cpts['Xray']('TuberculosisOrCancer_1') )

"""


asia_joint = reorder_state_domains(flatten(asia_graph,
                                           asia_cpts), 
                                   ['VisitToAsia', 
                                    'Tuberculosis', 
                                    'TuberculosisOrCancer', 
                                    'Xray', 
                                    'Dyspnea',
                                    'LungCancer',
                                    'Bronchitis',
                                    'Smoker'])

print("\n* Asia update 1")

asia_domain = asia_joint.dom

p = point_pred('Tuberculosis_0', asia_domain[1])

p1 = truth(asia_domain[0]) @ p @ truth(asia_domain[2:])

print( asia_joint / p1 % [0,0,0,0,1,0,0,0] )

print( asia_inference.query(['Dyspnea'], evidence={'Tuberculosis': 0})['Dyspnea'] )

print( asia_cpts['Dyspnea'] \
       >> ((asia_cpts['Bronchitis'] @ asia_cpts['TuberculosisOrCancer']) \
           >> ((idn(asia_domain[7]) @ asia_cpts['LungCancer'] @ idn(asia_domain[1])) \
               >> ((copy(asia_domain[7]) @ idn(asia_domain[1])) \
                   >> (asia_cpts['Smoker'] @ ((asia_cpts['Tuberculosis'] >> asia_cpts['VisitToAsia']) / p))))) )

print("\n* Asia update 2")

q = point_pred('Xray_1', asia_domain[3])

q1 = truth(asia_domain[0:3]) @ q @ truth(asia_domain[4:])

print( asia_joint / q1 % [0,0,0,0,0,0,1,0] )

print( asia_inference.query(['Bronchitis'], evidence={'Xray': 1})['Bronchitis'] )

print( asia_cpts['Bronchitis'] 
       >> ((asia_cpts['Smoker'] @ asia_cpts['VisitToAsia']) / ((asia_cpts['LungCancer'] @ asia_cpts['Tuberculosis']) \
      << (asia_cpts['TuberculosisOrCancer'] << (asia_cpts['Xray'] << q))) % [1,0]) )

print("\nTimed")

print( timeit.timeit(lambda: asia_inference.query(['Bronchitis'], evidence={'Xray': 1})['Bronchitis'],
                     number=N) )

print( timeit.timeit(lambda: asia_cpts['Bronchitis'] 
       >> ((asia_cpts['Smoker'] @ asia_cpts['VisitToAsia']) / ((asia_cpts['LungCancer'] @ asia_cpts['Tuberculosis']) \
      << (asia_cpts['TuberculosisOrCancer'] << (asia_cpts['Xray'] << q))) % [1,0]),
                    number=N) )

"""


#print("\nExperiments")
#print("===========\n")








#print("\nStudent stretch output\n")

#student_stretch = stretch(student_model)

#graph_image(student_stretch['graph'], "experiment2")

#print("\nStudent pointer:", student_stretch['pointer'] )
#print("\nStudent state:", evaluate_stretch(student_stretch['channels']) )

#print("\nStudent marginals:")
#marginals_stretch(student_stretch)

#print("\nStudent inference:")
#print( student_inference.query(['L'], evidence={'I': 0, 'D' : 1})['L'] )
#print( inference_query(student_stretch, 'L', {'I' : [1,0], 'D' : [0,1]}) )

# t1 = timeit.timeit(lambda: student_inference.query(['L'], evidence={'I': 0, 'D' : 1})['L'],
#                    number = N)

# t2 = timeit.timeit(lambda: inference_query(student_stretch, 'L', {'I' : [1,0], 'D' : [0,1]}),
#                    number = N)
# print(t1, t2, t1/t2)


# print( student_inference.query(['L'])['L'] )
# print( student_inference.query(['G'])['G'] )
# print( student_inference.query(['S'])['S'] )

print("\nAsia stretch output\n")

print(timeit.timeit(lambda: stretch(asia_model, graph_output=False), number=1))

asia_stretch = stretch(asia_model, graph_output=True,observed=False)

graph_image(asia_stretch['graph'], "experiment2")

#print("\nAsia pointer:", asia_stretch['pointer'] )
asia_joint = evaluate_stretch(asia_stretch['channels'])
#print("\nAsia state:", asia_joint )

#print("\nAsia marginals:"); marginals_stretch(asia_stretch)

# print( asia_inference.query(['Tuberculosis'])['Tuberculosis'] )
# print( asia_inference.query(['LungCancer'])['LungCancer'] )
# print( asia_inference.query(['Bronchitis'])['Bronchitis'] )
# print( asia_inference.query(['TuberculosisOrCancer'])['TuberculosisOrCancer'] )
# print( asia_inference.query(['Xray'])['Xray'] )
# print( asia_inference.query(['Dyspnea'])['Dyspnea'] )

# print("\nAsia inference:")

# print( asia_inference.query(['Bronchitis'], evidence={'Xray': 0, 'Smoker' : 1})['Bronchitis'] )
# print( inference_query(asia_stretch, 'Bronchitis', {'Xray' : [1,0], 'Smoker' : [0,1]}) )

# asia_joint = reorder_state_domains(asia_joint,
#                                    ['VisitToAsia', 
#                                     'Tuberculosis', 
#                                     'TuberculosisOrCancer', 
#                                     'Xray', 
#                                     'Dyspnea',
#                                     'LungCancer',
#                                     'Bronchitis',
#                                     'Smoker'])

# asia_domain = asia_joint.dom

# p = point_pred('Xray_0', asia_domain[3])

# p1 = truth(asia_domain[:3]) @ p @ truth(asia_domain[4:])

# q = point_pred('Smoker_1', asia_domain[7])

# q1 = truth(asia_domain[:7]) @ q

# print( asia_joint / (p1 & q1) % [0,0,0,0,0,0,1,0] )


# t1 = timeit.timeit(lambda: asia_inference.query(['Bronchitis'], evidence={'Xray': 0, 'Smoker' : 1})['Bronchitis'],
#                    number = N)
# t2 = timeit.timeit(lambda: inference_query(asia_stretch, 'Bronchitis', {'Xray' : [1,0], 'Smoker' : [0,1]}),
#                    number = N) 

# print(t1, t2, t1/t2)




