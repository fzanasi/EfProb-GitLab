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

sac_model = BayesianModel([('Smoking', 'Ashtray'), 
                           ('Smoking', 'Cancer')])

cpd_smoking = TabularCPD(variable='Smoking', variable_card=2,
                         values=[[0.3], [0.7]])

cpd_ashtray = TabularCPD(variable='Ashtray', variable_card=2,
                      values=[[0.95, 0.25], [0.05, 0.75]],
                      evidence=['Smoking'], evidence_card=[2])

cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.4, 0.05], [0.6, 0.95]],
                        evidence=['Smoking'], evidence_card=[2])

sac_model.add_cpds(cpd_smoking, cpd_ashtray, cpd_cancer)

print("\nSAC model correct: ", sac_model.check_model() )
print ("\nNodes: ", sac_model.nodes())
print("\nTables:")
print (sac_model.get_cpds('Smoking'))
print (sac_model.get_cpds('Ashtray'))
print (sac_model.get_cpds('Cancer'))

sac_inference = VariableElimination(sac_model)

print("\nMarginals:")
print( sac_inference.query(['Smoking']) ['Smoking'])
print( sac_inference.query(['Cancer']) ['Cancer'])
print( sac_inference.query(['Ashtray']) ['Ashtray'])

print("\nConditionals (note: 0 stands for true-case)")
print( sac_inference.query(['Cancer'], evidence={'Smoking': 0}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Smoking': 1}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Ashtray': 0}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Ashtray': 1}) ['Cancer'])


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

def efprob_domain(name, card):
    items = [ name + "_" + str(i) for i in range(card)]
    return Dom(items, names = Name(name))
    
def efprob_domains_of_pgm(model):
    nodes = model.nodes()
    domains = []
    for n in nodes:
        domains.append(efprob_domain(n, model.get_cardinality(n))) 
    return domains

def efprob_channels_of_pgm(model):
    nodes = model.nodes()
    cpts = {}
    for n in nodes:
        #print("Handling node: ", n)
        cpd = model.get_cpds(node=n)
        card_n = cpd.variable_card
        # first extract codomain
        cod = efprob_domain(cpd.variable, card_n)
        # double check: is reverse really neede here?
        reverse_parent_nodes = cpd.get_evidence()
        parent_nodes = [reverse_parent_nodes[i] for i in range(len(reverse_parent_nodes)-1, -1, -1)]
        # print(n, cpd, parent_nodes)
        parents = len(parent_nodes)
        vals = cpd.get_values() 
        #print(vals, vals.shape, card_n)
        if parents == 0:
            # include a state, not a channel, since there is no domain
            cpts[n] = State(vals.transpose()[0], cod)
        else:
            # extract a channel; first find its domain
            doms = [efprob_domain(pn, model.get_cardinality(pn)) 
                    for pn in parent_nodes]
            dom = reduce(lambda d1, d2: d1 + d2, doms)
            states = []
            for i in range(parents):
                parent_i = parent_nodes[i]
                card_i = model.get_cardinality(parent_i)
                states = states + [State(vals.transpose()[j + card_i*i], cod) 
                                   for j in range(card_i)]
            #print(states)
            cpts[n] = chan_from_states(states, dom)
    return cpts

def pydot_graph_of_pgm(model):
    graph = pydot.Dot(graph_type='digraph')
    nodes = model.nodes()
    for n in nodes:
        graph.add_node(pydot.Node(n))
    edges = model.edges()
    for e in edges:
        graph.add_edge(pydot.Edge(e))
    return graph


#print("\nChannels\n")

#print( efprob_channel_from_tabularCPD(cpd_ashtray) )
#print( efprob_channel_from_tabularCPD(cpd_g) )
#print( efprob_channel_from_tabularCPD(cpd_l) )

#print("\nGraphs:")

#sac_graph = pydot_graph_of_pgm(sac_model)

#graph_image(sac_graph, "sac")

#sac_cpts = efprob_channels_of_pgm(sac_model)

# print( sac_cpts["Smoking"] )
# print( sac_cpts["Ashtray"] )
# print( sac_cpts["Ashtray"]("Smoking_0") )
# print( sac_cpts["Ashtray"]("Smoking_1") )
# print( sac_cpts["Cancer"] )
# print( sac_cpts["Cancer"]("Smoking_0") )
# print( sac_cpts["Cancer"]("Smoking_1") )

#print( flatten(sac_graph, sac_cpts) )

#0.114|A,C,S> + 0.00875|A,C,~S> + 0.171|A,~C,S> + 0.166|A,~C,~S> + 0.006|~A,C,S> + 0.0263|~A,C,~S> + 0.009|~A,~C,S> + 0.499|~A,~C,~S>

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


def node_num(node, num):
    return node + "!" + str(num)

#
# Turn a Probabilistic Graphical Model (pgm, as in pgmpy) into a
# sequence of channels, together with a node_pointer dictionary which
# tells for each node in the model where it occurs in this sequence.
#
# Optionally, via additional True/False settings
#
# - a pydot graph is produced of the pgm (with additional copy and
#   identity nodes), indicating the burocratic channels for copying
#   and reordering
#
# - an 'observed' version, both of the sequence of channels and of the
#   graph, is produced, in which internal nodes become 'observable'
#   (or accessible). This version is useful for producing the joint
#   distribution from the model.
#
def stretch(pgm, graph_output=True, observed=False):
    #
    # Extraction of relevant data from input
    #
    nodes = pgm.nodes
    domains = efprob_domains_of_pgm(pgm)
    channels = efprob_channels_of_pgm(pgm)
    parents = {}
    children = {}
    for n in pgm.nodes():
        parents[n] = []
        children[n] = []
    for e in pgm.edges():
        children[e[0]].append(e[1])
        parents[e[1]].append(e[0])
    #
    # Initialisation of data structures that will be built up:
    # (a) list of channels, (b) nodes pointers, (c) graph, optionally
    #
    # (1) List of channels, where the first entry consists of the
    # (product of the) initial states.
    #
    channel_list = []
    # 
    # (b) Dictionary with index level (natural number) for each node
    #
    node_pointer = {}
    pointer = 2
    #
    # (c) Optional graph of the linearised graph; original nodes are
    # in green
    #
    if graph_output:
        stretched_graph = pydot.Dot(graph_type='digraph')
    #
    # List of available nodes for finding a match of
    # channels. Elements of this list will be pairs, consisting of the
    # node (name) 'n' together with its name node_name(n,i) in the
    # graph. Since nodes may occur multiple times, we need to use
    # different names in the graph.
    #
    available_nodes = []
    #
    # Step 1: handling of initial nodes; 
    #
    # deep copy
    unprocessed_nodes = [n for n in nodes]
    initial_nodes = []
    initial_copy_channels = []
    for n in nodes:
        # print("Handling node: ", n)
        parents_n = parents[n]
        if len(parents_n) == 0:
            # n is an initial node
            initial_nodes.append(n)
            unprocessed_nodes.remove(n)
            node_pointer[n] = 0          
            if graph_output:
                stretched_graph.add_node(pydot.Node(n, 
                                                    style="filled", 
                                                    fillcolor="green"))
            if observed:
                available_nodes.append(n)
            children_n = children[n] 
            children_num = len(children_n)
            # print("Children of: ", n, children_n)
            #
            # Make copies of the initial states, depending on the
            # number of children (and on whether or not we want an
            # 'observed' version)
            #
            if children_num > 0:
                if graph_output:
                    stretched_graph.add_node(pydot.Node(n + "!copy", 
                                                        width=0.15,
                                                        height=0.15,
                                                        fixedsize=True,
                                                        style="filled", 
                                                        fillcolor="black",
                                                        fontsize=0))
                    stretched_graph.add_edge(pydot.Edge(n, n + "!copy"))
                    if observed:
                        stretched_graph.add_edge(pydot.Edge(n + "!copy",
                                                            " " + n + " "))
                available_nodes += children_num * [n]
                initial_copy_channels.append(copy(channels[n].dom, 
                                                  children_num + 
                                                  (1 if observed else 0)))
    if len(initial_nodes) == 0:
        raise Exception('Error: the model does not have initial nodes')
    state = reduce(lambda s1, s2: s1 @ s2, [channels[n]
                                            for n in initial_nodes])
    channel_list.append(state)
    if len(initial_copy_channels) == 0:
        # Initial nodes have no children; extremely simple graph
        channel_list.append(idn(state))
    else:
        initial_copy_chan = reduce(lambda c1, c2: c1 @ c2, 
                                   initial_copy_channels)
        channel_list.append(initial_copy_chan)
    #print("Initial state: ", channel_list[1] >> state )
    #
    # Step 2: continue with non-initial nodes from the model: cycle
    # through the set of unprocessed nodes until all associated
    # channels have been applied to the current state, and the list of
    # unprocessed nodes is empty.
    #
    iterations = 0
    while len(unprocessed_nodes) > 0:
        for un in unprocessed_nodes:
            iterations += 1
            print("* Iteration:", iterations, un)
            un_chan = channels[un]
            parents_un = [n.name for n in un_chan.dom.names]
            num_parents_un = len(parents_un)
            #print("Parents of: ", un, parents_un)
            search_copy_of_nodes = [u for u in available_nodes]
            swaps = list(range(len(available_nodes)))
            #
            # find occurrences of un's parents in domains
            #
            i = 0
            found_all = True
            while i < num_parents_un:
                #print("... searching for parent: ", parents_un[i] )
                #
                # try to find i-th parent among domains
                #
                j = 0
                found_i = False
                while j < len(available_nodes):
                    if search_copy_of_nodes[j] == parents_un[i]:
                        found_i = True
                        break
                    j += 1
                if not found_i:
                    #
                    # stop handling parent i
                    #
                    found_all = False
                    #print("Stop handling node: ", un)
                    break
                #
                # i-th parent found at j
                #
                #print("=> Parent found of:", un, "=", parents_un[i], "at", j)
                #
                # swap j |-> i
                #
                swaps[j] = swaps[i]
                swaps[i] = j
                search_copy_of_nodes[j] = search_copy_of_nodes[i]
                search_copy_of_nodes[i] = parents_un[i]
                # print("Search copy: ", search_copy_of_nodes)
                i += 1
            if found_all:
                #
                # all parents found; now update the state with channel of un
                #
                #print("==> All parents found of:", un)
                #print("Available domains: ", available_nodes)
                if graph_output:
                    stretched_graph.add_node(pydot.Node(un, 
                                                        style="filled", 
                                                        fillcolor="green"))
                #print("Swaps:", swaps )
                #
                # incorporate swaps into available nodes and arguments
                #
                swaps_len = len(swaps)
                argument_swaps = list(range(swaps_len))
                for i in range(num_parents_un):
                    tmp = available_nodes[i]
                    available_nodes[i] = available_nodes[swaps[i]]
                    available_nodes[swaps[i]] = tmp
                    tmp = argument_swaps[i]
                    argument_swaps[i] = argument_swaps[swaps[i]]
                    argument_swaps[swaps[i]] = tmp
                    if graph_output:
                        stretched_graph.add_edge(pydot.Edge(
                            available_nodes[i] + "!copy", un))
                #print("Swapped domains: ", available_nodes)
                #
                # Build the channel that does the swapping
                #
                current_dom = channel_list[len(channel_list)-1].cod
                swapped_doms = []
                for i in range(swaps_len):
                    swapped_doms.append(current_dom.get_nameditem(
                        argument_swaps[i]))
                swapped_dom = reduce(lambda d1, d2: d1 + d2, swapped_doms)
                swap_chan = chan_fromklmap(lambda *xs: 
                                           point_state(
                                               tuple([xs[argument_swaps[i]] 
                                                      for i in range(swaps_len)]),
                                               swapped_dom),
                                           current_dom, swapped_dom)
                diff = len(available_nodes) - num_parents_un
                un_chan_id = un_chan
                identities = None
                if diff > 0:
                    identities_doms = []
                    for i in range(diff):
                        d_i = swapped_dom.get_nameditem(i + num_parents_un)
                        identities_doms.append(d_i)
                    identities_dom = reduce(lambda d1, d2: d1 + d2, 
                                            identities_doms)
                    identities = idn(identities_dom)
                    un_chan_id = un_chan @ identities
                #
                # Add un_chan, extended with identities and preceded
                # by swaps, to the list of channels
                #
                channel_list.append(un_chan_id * swap_chan)
                pointer += 1
                node_pointer[un] = pointer
                #
                # Update the available nodes
                #
                tails = available_nodes[num_parents_un:]
                heads = [un]
                childred_un = children[un] 
                num_children_un = len(childred_un)
                #print("Children of: ", un, childred_un)
                if num_children_un > 0:
                    if graph_output:
                        stretched_graph.add_node(pydot.Node(un + "!copy", 
                                                            width=0.15,
                                                            height=0.15,
                                                            fixedsize=True,
                                                            style="filled", 
                                                            fillcolor="black",
                                                            fontsize=0))
                        stretched_graph.add_edge(pydot.Edge(un, un + "!copy"))
                        if observed:
                            stretched_graph.add_edge(pydot.Edge(un + "!copy",
                                                                " " + un + " "))
                    if num_children_un > 1 or observed:
                        heads += (num_children_un - 1 + 
                                  (1 if observed else 0)) * [un]
                        copies = copy(un_chan.cod, 
                                      num_children_un + (1 if observed else 0))
                        if diff > 0:
                            copies = copies @ identities
                        channel_list.append(copies)
                        pointer += 1
                available_nodes = heads + tails
                #
                # Terminate handling node un and return to for-loop
                #
                unprocessed_nodes.remove(un)
    #
    # Collect the results in a dictionary
    # 
    result = {
        'pointer'  : node_pointer,
        'channels' : channel_list
    }
    if graph_output:
        result['graph'] = stretched_graph
    return result


def evaluate_stretch(chan_list):
    state = chan_list[0]
    #print("Length channel list:", len(chan_list))
    for i in range(len(chan_list)-1):
        #print("Eval:", i, state.dom, state)
        state = chan_list[i+1] >> state
    return state

def marginals_stretch(stretch_dict):
    cl = stretch_dict['channels']
    pr = stretch_dict['pointer']
    for n in pr.keys():
        # print(n, pr[n])
        if pr[n] > 0:
            state = evaluate_stretch(cl[0:pr[n]])
            #print(n, pr[n], state.dom )
            mask = len(state.dom) * [0]
            mask[0] = 1
            print(n, "->", state % mask )
    return None

#
# This function operates on the output of the 'stretch' function and
# uses it for inference. It takes two additional arguments:
#
# - 'marginal', which is a node in the original model, at which
#   position the probability is computed, given the evidence
#
# - 'evidence_dict', which is a dictionary mapping nodes to list of
#   probabilities in [0,1], acting as predicate values. If the node
#   has cardinality n, the list must have length n.
#
def inference_query(stretch_dict, marginal, evidence_dict):
    chan_list = stretch_dict['channels']
    chan_list_len = len(chan_list)
    state = chan_list[0]
    init_pred = truth(state.dom)
    ptr = stretch_dict['pointer']
    ptrkeys = ptr.keys()
    evkeys = evidence_dict.keys()
    if not (marginal in ptrkeys):
        raise Exception('Marginal does not occur')
    if not any([k in ptrkeys for k in evkeys]):
        raise Exception('Some of the evidence keys do not occur')
    #
    # The evidence is inserted into a list in parallel to the list of
    # channels, so that updates can happen locally. This evidence-list
    # contains two kinds of entries: 
    #
    # - (0,None) --- nothing will happen here 
    # - (1,pred)
    #
    evidence_list = chan_list_len * [(0,None)]
    for k in evkeys:
        position_k = ptr[k]
        if position_k == 0:
            #
            # evidence at an initial node; the state is then updated
            # immediately with the corresponding (weakened) predicate;
            # nothing is added to the evidence list.
            #
            dom = state.dom
            preds = []
            for i in range(len(dom)):
                if k == dom.names[i].name:
                    preds.append(Predicate(evidence_dict[k], dom[i]))
                else:
                    preds.append(truth(dom[i]))
            weakened_pred = reduce(lambda p1, p2: p1 @ p2, preds)
            state = state / weakened_pred
        else:
            dom = chan_list[position_k-1].cod
            #print("Evidence at:", k, position_k, chan_list_len, dom)
            #
            # By construction, the domain of the predicate is at the
            # first position in the codomain of channel at this
            # position
            #
            pred = Predicate(evidence_dict[k], dom[0])
            if len(dom) > 0:
                # weaken
                pred = pred @ truth(dom[1:])
            evidence_list[position_k - 1] = (1,pred)
    position_marginal = ptr[marginal]
    #
    # Start with forward state transformation up to the marginal
    # position, updating with predicates, if any, along the way
    #
    for i in range(position_marginal-1):
        if evidence_list[i][0] == 0:
            # no predicate at this point, transform existing state
            state = chan_list[i+1] >> state
        else:
            # update the current state with the predicate first, then
            # transform, i.e. do forward inference
            state = state / evidence_list[i][1]
            state = chan_list[i+1] >> state
    #
    # Continue with backward predicate transformation, starting from
    # the truth predicate, adding predicates, if any, along the way,
    # via conjunction &
    #
    pred = truth(chan_list[chan_list_len-1].cod)
    for i in range(chan_list_len - position_marginal):
        if evidence_list[chan_list_len - i - 1][0] == 0:
            # no predicate at this point, transform existing predicate
            pred = chan_list[chan_list_len - i - 1] << pred
        else:
            # take the conjunction of the current predicate with the
            # predicate at this point, then transform
            pred = pred & evidence_list[chan_list_len - i - 1][1]
            pred = chan_list[chan_list_len - i - 1] << pred
    #
    # Combine the results of the forward and backward operations by
    # updating the state once more, and taking the appropriate (first)
    # marginal.
    #
    state = state / pred
    mask = len(state.dom) * [0]
    mask[0] = 1
    return state % mask


#print("SAC stretch output\n")

#sac_stretch = stretch(sac_model)

#graph_image(sac_stretch['graph'], "experiment1")

#print("\nSAC pointer:", sac_stretch['pointer'] )
#print("\nSAC state:", evaluate_stretch(sac_stretch['channels']) )

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

asia_stretch = stretch(asia_model, graph_output=True,observed=True)

#graph_image(asia_stretch['graph'], "experiment2")

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




