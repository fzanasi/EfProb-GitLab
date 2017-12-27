from efprob_dc import *
from baynet import *

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

print("")

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
sac_cpts = factorise(sac_joint, sac_graph)
print("Distributions obtained from the Bayesian network:")
print( sac_cpts["smoking"] )
print( sac_cpts["ashtray"]('S'), " and ", sac_cpts["ashtray"]('~S') )
print( sac_cpts["cancer"]('S'), " and ", sac_cpts["cancer"]('~S') )

#print("\nReconstructed state after flattening sac")

#reconstructed_sac = flatten(sac_graph, sac_cpts)
#print( reconstructed_sac )

#reordered_reconstructed_sac = reorder_state_domains(reconstructed_sac, 
#                                                    sac_joint.dom)
#print("\nReconstruction equal to orginal: ", 
#      reordered_reconstructed_sac == sac_joint )
print("Match distance: ", state_graph_match(sac_joint, sac_graph))


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
visit_joint = \
  ((idn(asia_dom) @ idn(tuberculosis_dom) @ idn(disjunction_dom) @ xray @ dyspnoea @ idn(cancer_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ idn(tuberculosis_dom) @ copy(disjunction_dom,3) @ idn(bronchitis_dom) @ idn(cancer_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ idn(tuberculosis_dom) @ disjunction @ swap(cancer_dom,bronchitis_dom) @ idn(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ copy(tuberculosis_dom) @ copy(cancer_dom) @ copy(bronchitis_dom) @ idn(smoking_dom)) \
   * (idn(asia_dom) @ tuberculosis @ cancer @ bronchitis @ idn(smoking_dom)) \
   * (copy(asia_dom) @ copy(smoking_dom,3))) \
   >> asia @ smoking


print( visit_joint )

#print( visit_joint % [0,0,0,1,1,0,0,0] )

"""


visit_graph = pydot.Dot(graph_type='digraph')
#
visit_graph.add_node(pydot.Node("asia"))
visit_graph.add_node(pydot.Node("tuberculosis"))
visit_graph.add_node(pydot.Node("disjunction"))
visit_graph.add_node(pydot.Node("xray"))
visit_graph.add_node(pydot.Node("dyspnoea"))
visit_graph.add_node(pydot.Node("cancer"))
visit_graph.add_node(pydot.Node("bronchitis"))
visit_graph.add_node(pydot.Node("smoking"))
#
visit_graph.add_edge(pydot.Edge("asia", "tuberculosis"))
visit_graph.add_edge(pydot.Edge("tuberculosis", "disjunction"))
visit_graph.add_edge(pydot.Edge("disjunction", "xray"))
visit_graph.add_edge(pydot.Edge("disjunction", "dyspnoea"))
visit_graph.add_edge(pydot.Edge("smoking", "cancer"))
visit_graph.add_edge(pydot.Edge("smoking", "bronchitis"))
visit_graph.add_edge(pydot.Edge("cancer", "disjunction"))
visit_graph.add_edge(pydot.Edge("bronchitis", "dyspnoea"))

#
# Save and display graph
#
#graph_image(visit_graph, "visit")

# visit_cpts = factorise(visit_joint, visit_graph)
# reconstructed_visit_state = flatten(visit_graph, visit_cpts)

# reordered_reconstructed_visit_state = \
#     reorder_state_domains(reconstructed_visit_state, visit_joint.dom)

# print( visit_joint )
# print( reconstructed_visit_state )
# print( reordered_reconstructed_visit_state )

# print("Properly reconstructed visit: ", 
#       reordered_reconstructed_visit_state == visit_joint )

print("\nVisit match: ", state_graph_match(visit_joint, visit_graph))




print("\nFour node test")
print("==============\n")

A = Dom(['a','~a'], names = Name("A"))
B = Dom(['b','~b'], names = Name("B"))
C = Dom(['c','~c'], names = Name("C"))
D = Dom(['d','~d'], names = Name("D"))

CD_graph = pydot.Dot(graph_type='digraph')
CD_graph.add_node(pydot.Node("A"))
CD_graph.add_node(pydot.Node("B"))
CD_graph.add_node(pydot.Node("C"))
CD_graph.add_node(pydot.Node("D"))
CD_graph.add_edge(pydot.Edge("A", "C"))
CD_graph.add_edge(pydot.Edge("B", "C"))
CD_graph.add_edge(pydot.Edge("C", "D"))

DC_graph = pydot.Dot(graph_type='digraph')
DC_graph.add_node(pydot.Node("A"))
DC_graph.add_node(pydot.Node("B"))
DC_graph.add_node(pydot.Node("C"))
DC_graph.add_node(pydot.Node("D"))
DC_graph.add_edge(pydot.Edge("A", "C"))
DC_graph.add_edge(pydot.Edge("B", "C"))
DC_graph.add_edge(pydot.Edge("D", "C"))

#
# Save and display graph
#
#graph_image(CD_graph, "cd")
#graph_image(DC_graph, "dc")


random_state = random_state(A + B + C + D)

print("Random state with four nodes:\n", random_state )

print("\nTwo different match computations")
print("CD match: ", state_graph_match(random_state, CD_graph))
print("DC match: ", state_graph_match(random_state, DC_graph))
                                                           

##############################################
#
# Discrete Weather-Play, see also:
# http://arxiv.org/abs/1709.00322
#
##############################################

print("\nDiscrete Weather-Play")
print("=====================\n")

Outlook = Dom(['S', 'O', 'R'], names = Name("Outlook"))
Temp = Dom(['H', 'M', 'C'], names = Name("Temp"))
Humidity = Dom(['H', 'N'], names = Name("Humidity"))
Windy = Dom(['t', 'f'], names = Name("Windy"))
Play = Dom(['y', 'n'], names = Name("Play"))

D = Outlook + Temp + Humidity + Windy + Play

table = 1/14  *  point_state(('S', 'H', 'H', 'f', 'n'), D) \
        + 1/14 * point_state(('S', 'H', 'H', 't', 'n'), D) \
        + 1/14 * point_state(('O', 'H', 'H', 'f', 'y'), D) \
        + 1/14 * point_state(('R', 'M', 'H', 'f', 'y'), D) \
        + 1/14 * point_state(('R', 'C', 'N', 'f', 'y'), D) \
        + 1/14 * point_state(('R', 'C', 'N', 't', 'n'), D) \
        + 1/14 * point_state(('O', 'C', 'N', 't', 'y'), D) \
        + 1/14 * point_state(('S', 'M', 'H', 'f', 'n'), D) \
        + 1/14 * point_state(('S', 'C', 'N', 'f', 'y'), D) \
        + 1/14 * point_state(('R', 'M', 'N', 'f', 'y'), D) \
        + 1/14 * point_state(('S', 'M', 'N', 't', 'y'), D) \
        + 1/14 * point_state(('O', 'M', 'H', 't', 'y'), D) \
        + 1/14 * point_state(('O', 'H', 'N', 'f', 'y'), D) \
        + 1/14 * point_state(('R', 'M', 'H', 't', 'n'), D) 

#
# Experiment with various edges in order to get the least
# state-graph-match value
#
weather_play_graph = pydot.Dot(graph_type='digraph')
weather_play_graph.add_node(pydot.Node("Outlook"))
weather_play_graph.add_node(pydot.Node("Temp"))
weather_play_graph.add_node(pydot.Node("Humidity"))
weather_play_graph.add_node(pydot.Node("Windy"))
weather_play_graph.add_node(pydot.Node("Play"))
# Outlook
weather_play_graph.add_edge(pydot.Edge("Outlook","Temp"))
#weather_play_graph.add_edge(pydot.Edge("Outlook","Humidity"))
#weather_play_graph.add_edge(pydot.Edge("Outlook","Windy"))
weather_play_graph.add_edge(pydot.Edge("Outlook","Play"))
# Temp
#weather_play_graph.add_edge(pydot.Edge("Temp","Outlook"))
weather_play_graph.add_edge(pydot.Edge("Temp","Humidity"))
weather_play_graph.add_edge(pydot.Edge("Temp","Windy"))
#weather_play_graph.add_edge(pydot.Edge("Temp","Play"))
# Humidity
#weather_play_graph.add_edge(pydot.Edge("Humidity", "Outlook"))
#weather_play_graph.add_edge(pydot.Edge("Humidity", "Temp"))
#weather_play_graph.add_edge(pydot.Edge("Humidity", "Windy"))
weather_play_graph.add_edge(pydot.Edge("Humidity", "Play"))
# Windy
#weather_play_graph.add_edge(pydot.Edge("Windy", "Outlook"))
#weather_play_graph.add_edge(pydot.Edge("Windy", "Temp"))
#weather_play_graph.add_edge(pydot.Edge("Windy", "Humidity"))
weather_play_graph.add_edge(pydot.Edge("Windy", "Play"))
# Play
#weather_play_graph.add_edge(pydot.Edge("Play", "Outlook"))
#weather_play_graph.add_edge(pydot.Edge("Play", "Temp"))
#weather_play_graph.add_edge(pydot.Edge("Play", "Humidity"))
#weather_play_graph.add_edge(pydot.Edge("Play", "Windy"))

#
# Save and display graph
#
#graph_image(weather_play_graph, "weather")

print("Weather-play match: ", state_graph_match(table, weather_play_graph))


"""
