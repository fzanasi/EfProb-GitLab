#
# model class code at:
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/models/BayesianModel.py
#
# TabularCPD in:
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/CPD.py
#
from pgm_efprob import *
#from efprob_dc import *
#from baynet import *

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
#from pgmpy.inference import VariableElimination
#from pgmpy.inference import BeliefPropagation

#import pydot
#from PIL import Image
#import timeit


##############################################
#
# sac = smoking-ashtray-cancer
#
#    ashtray   cancer
#       /\      /\
#        \      /
#         \    /
#        smoking
#
##############################################

print("\nSmoking-Ashtray-Cancer example")
print("==============================\n")

#
# Named domains
#
sd = Dom(['S', '~S'], names = Name("smoking"))
ad = Dom(['A', '~A'], names = Name("ashtray"))
cd = Dom(['C', '~C'], names = Name("cancer"))

#
# Initial state and channels
#
smoking = flip(0.3, sd)
ashtray = chan_from_states([flip(0.95,ad), flip(0.25,ad)], sd)
cancer = chan_from_states([flip(0.4,cd), flip(0.05,cd)], sd) 

sac_joint = (ashtray @ cancer @ idn(sd)) * copy(sd,3) >> smoking

print( sac_joint )

#
# Graph 
#
sac_graph = pydot.Dot(graph_type='digraph')
# nodes, first defined explicitly
ash = pydot.Node("ashtray", style="filled", fillcolor="green")
ca = pydot.Node("cancer", style="filled", fillcolor="red")
sm = pydot.Node("smoking", style="filled", fillcolor="yellow")
sac_graph.add_node(ash)
sac_graph.add_node(sm)
sac_graph.add_node(ca)
# edges, first defined explicitly
sm_ash = pydot.Edge(sm, ash)
sm_ca = pydot.Edge(sm, ca)
sac_graph.add_edge(sm_ash)
sac_graph.add_edge(sm_ca)
#sac_graph.add_edge(pydot.Edge("ashtray", "cancer"))

#
# Save and display graph
#
#graph_image(sac_graph, "sac")

#
# Form Bayesian network
#
sac_model = factorise(sac_joint, sac_graph)

sac_model_graph = pydot_graph_of_pgm(sac_model)
#graph_image(sac_model_graph, "sac")

print( sac_model.get_cpds(node='smoking') )
print( sac_model.get_cpds(node='ashtray') )
print( sac_model.get_cpds(node='cancer') )


##############################################
#
# Asia visit example
#
# From: http://www.cse.sc.edu/~mgv/talks/AIM2010.ppt
#
#   asia --> tuberculosis    xray
#                 \        _
#                 _\|       /|
#                          /
#      cancer --> disjunction --> dyspnoea
#       _                          _
#        /|                        /|
#       /                         /
#   smoking ---------------> bronchitis
#
##############################################

print("\nAsia visit example")
print("==================\n")

#
# Named domains
#
asia_dom = Dom(['A', '~A'], names = Name("asia"))
smoking_dom = Dom(['S', '~S'], names = Name("smoking"))
tuberculosis_dom = Dom(['T', '~T'], names = Name("tuberculosis"))
disjunction_dom = Dom(['O', '~O'], names = Name("disjunction"))
cancer_dom = Dom(['C', '~C'], names = Name("cancer"))
xray_dom = Dom(['X', '~X'], names = Name("xray"))
dyspnoea_dom = Dom(['D', '~D'], names = Name("dyspnoea"))
bronchitis_dom = Dom(['B', '~B'], names = Name("bronchitis"))

#
# Initial states
#
asia = flip(0.01, asia_dom) 
smoking = flip(0.5, smoking_dom) 

#
# Channels
#
tuberculosis = chan_from_states([flip(0.05, tuberculosis_dom),
                                 flip(0.01, tuberculosis_dom)], 
                                asia_dom)

# disjunction = chan_from_states([flip(0.9, disjunction_dom),
#                                 flip(0.7, disjunction_dom),
#                                 flip(0.7, disjunction_dom),
#                                 flip(0.1, disjunction_dom)],
#                                tuberculosis_dom + cancer_dom)

disjunction = chan_from_states([flip(1, disjunction_dom),
                                flip(0, disjunction_dom),
                                flip(0, disjunction_dom),
                                flip(0, disjunction_dom)],
                               tuberculosis_dom + cancer_dom)

cancer = chan_from_states([flip(0.1, cancer_dom),
                           flip(0.01, cancer_dom)],
                          smoking_dom)

xray = chan_from_states([flip(0.98, xray_dom),
                         flip(0.05, xray_dom)],
                        disjunction_dom)

dyspnoea = chan_from_states([flip(0.9, dyspnoea_dom),
                             flip(0.7, dyspnoea_dom),
                             flip(0.8, dyspnoea_dom),
                             flip(0.1, dyspnoea_dom)],
                            disjunction_dom + bronchitis_dom)

bronchitis = chan_from_states([flip(0.6, bronchitis_dom),
                               flip(0.3, bronchitis_dom)],
                              smoking_dom)


#
# Add wires from internal nodes to the outside, so that the 8 outgoing 
# wires are respectively:
# 1. asia
# 2. tuberculosis
# 3. disjunction
# 4. xray
# 5. dyspnoea
# 6. cancer
# 7. bronchitis
# 8. smoking
#
asia_joint = \
  ((idn(asia_dom) @ idn(tuberculosis_dom) @ idn(disjunction_dom) @ xray @ dyspnoea @ idn(cancer_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ idn(tuberculosis_dom) @ copy(disjunction_dom,3) @ idn(bronchitis_dom) @ idn(cancer_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ idn(tuberculosis_dom) @ disjunction @ swap(cancer_dom,bronchitis_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ copy(tuberculosis_dom) @ copy(cancer_dom) @ copy(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ tuberculosis @ cancer @ bronchitis @ idn(smoking_dom)) \
   * (copy(asia_dom) @ copy(smoking_dom,3))) \
   >> asia @ smoking


#print( visit_joint )


asia_graph = pydot.Dot(graph_type='digraph')
#
asia_graph.add_node(pydot.Node("asia"))
asia_graph.add_node(pydot.Node("tuberculosis"))
asia_graph.add_node(pydot.Node("disjunction"))
asia_graph.add_node(pydot.Node("xray"))
asia_graph.add_node(pydot.Node("dyspnoea"))
asia_graph.add_node(pydot.Node("cancer"))
asia_graph.add_node(pydot.Node("bronchitis"))
asia_graph.add_node(pydot.Node("smoking"))
#
asia_graph.add_edge(pydot.Edge("asia", "tuberculosis"))
asia_graph.add_edge(pydot.Edge("tuberculosis", "disjunction"))
asia_graph.add_edge(pydot.Edge("disjunction", "xray"))
asia_graph.add_edge(pydot.Edge("disjunction", "dyspnoea"))
asia_graph.add_edge(pydot.Edge("smoking", "cancer"))
asia_graph.add_edge(pydot.Edge("smoking", "bronchitis"))
asia_graph.add_edge(pydot.Edge("cancer", "disjunction"))
asia_graph.add_edge(pydot.Edge("bronchitis", "dyspnoea"))


# Save and display graph
#
#graph_image(asia_graph, "visit")


#
# Form Bayesian network
#
asia_model = factorise(asia_joint, asia_graph)

asia_model_graph = pydot_graph_of_pgm(asia_model)
#graph_image(asia_model_graph, "asia")

print( asia_model.get_cpds(node='smoking') )
print( asia_model.get_cpds(node='asia') )
print( asia_model.get_cpds(node='tuberculosis') )
print( asia_model.get_cpds(node='cancer') )
print( asia_model.get_cpds(node='bronchitis') )
print( asia_model.get_cpds(node='disjunction') )
print( asia_model.get_cpds(node='xray') )
print( asia_model.get_cpds(node='dyspnoea') )


asia_inference = VariableElimination(asia_model)

print( asia_inference.query(['dyspnoea'], 
                            evidence={'tuberculosis': 0})['dyspnoea'] )
