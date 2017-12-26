from efprob_dc import *
from baynet import *

#
# Experiments with 2, 3 and 4 nodes in a graph, and random states
# whose match values to various graphs are computed.
#
# One (experimental) finding is that Markov equivalent graph give the
# same match values. The underlying reason is that for Markov
# equivalent graphs the flattened state, obtained from a factorises
# state, is the same as the original. This has been checked for one
# examples in the 3-node case (together with Kenta).
#

A = Dom(['a','~a'], names = Name("A"))
B = Dom(['b','~b'], names = Name("B"))
C = Dom(['c','~c'], names = Name("C"))
D = Dom(['d','~d'], names = Name("D"))

AB_graph = pydot.Dot(graph_type='digraph')
AB_graph.add_node(pydot.Node("A"))
AB_graph.add_node(pydot.Node("B"))
#AB_graph.add_edge(pydot.Edge("A", "B"))

BA_graph = pydot.Dot(graph_type='digraph')
BA_graph.add_node(pydot.Node("A"))
BA_graph.add_node(pydot.Node("B"))
BA_graph.add_edge(pydot.Edge("B", "A"))

s2 = random_state(A @ B)

#
# A B are independent in the (uniform) state
# The graphs have less independence: hence this is an I-map: G subset D
#
su = uniform_state(A @ B)

#print( KLdivergence(s2, su) )

#graph_image(AB_graph); graph_image(BA_graph)

print("\n* Two-node case")
print( s2 )
print("AB match: ", state_graph_match(s2, AB_graph) )
print("BA match: ", state_graph_match(s2, BA_graph) )
print( su )
print("uniform AB match: ", state_graph_match(su, AB_graph) )
print("uniform BA match: ", state_graph_match(su, BA_graph) )

"""

ABC_graph = pydot.Dot(graph_type='digraph')
ABC_graph.add_node(pydot.Node("A"))
ABC_graph.add_node(pydot.Node("B"))
ABC_graph.add_node(pydot.Node("C"))
ABC_graph.add_edge(pydot.Edge("A", "B"))
ABC_graph.add_edge(pydot.Edge("B", "C"))

CBA_graph = pydot.Dot(graph_type='digraph')
CBA_graph.add_node(pydot.Node("A"))
CBA_graph.add_node(pydot.Node("B"))
CBA_graph.add_node(pydot.Node("C"))
CBA_graph.add_edge(pydot.Edge("B", "A"))
CBA_graph.add_edge(pydot.Edge("C", "B"))

BAC_graph = pydot.Dot(graph_type='digraph')
BAC_graph.add_node(pydot.Node("A"))
BAC_graph.add_node(pydot.Node("B"))
BAC_graph.add_node(pydot.Node("C"))
BAC_graph.add_edge(pydot.Edge("B", "A"))
BAC_graph.add_edge(pydot.Edge("B", "C"))

ACB_graph = pydot.Dot(graph_type='digraph')
ACB_graph.add_node(pydot.Node("A"))
ACB_graph.add_node(pydot.Node("B"))
ACB_graph.add_node(pydot.Node("C"))
ACB_graph.add_edge(pydot.Edge("A", "B"))
ACB_graph.add_edge(pydot.Edge("C", "B"))

AABC_graph = pydot.Dot(graph_type='digraph')
AABC_graph.add_node(pydot.Node("A"))
AABC_graph.add_node(pydot.Node("B"))
AABC_graph.add_node(pydot.Node("C"))
AABC_graph.add_edge(pydot.Edge("A", "B"))
AABC_graph.add_edge(pydot.Edge("A", "C"))
AABC_graph.add_edge(pydot.Edge("B", "C"))

#graph_image(ABC_graph); graph_image(CBA_graph); graph_image(BAC_graph); graph_image(ACB_graph)
#graph_image(AABC_graph)

s3 = random_state(A @ B @ C)

print("\n* Three-node case")
print( s3 )
print("ABC match: ", state_graph_match(s3, ABC_graph) )
print("CBA match: ", state_graph_match(s3, CBA_graph) )
print("BAC match: ", state_graph_match(s3, BAC_graph) )
print("ACB match: ", state_graph_match(s3, ACB_graph) )
#print("AABC match: ", state_graph_match(s3, AABC_graph) )


ABCD_graph = pydot.Dot(graph_type='digraph')
ABCD_graph.add_node(pydot.Node("A"))
ABCD_graph.add_node(pydot.Node("B"))
ABCD_graph.add_node(pydot.Node("C"))
ABCD_graph.add_node(pydot.Node("D"))
ABCD_graph.add_edge(pydot.Edge("A", "B"))
ABCD_graph.add_edge(pydot.Edge("A", "C"))
ABCD_graph.add_edge(pydot.Edge("B", "D"))
ABCD_graph.add_edge(pydot.Edge("C", "D"))

BACD_graph = pydot.Dot(graph_type='digraph')
BACD_graph.add_node(pydot.Node("A"))
BACD_graph.add_node(pydot.Node("B"))
BACD_graph.add_node(pydot.Node("C"))
BACD_graph.add_node(pydot.Node("D"))
BACD_graph.add_edge(pydot.Edge("B", "A"))
BACD_graph.add_edge(pydot.Edge("B", "D"))
BACD_graph.add_edge(pydot.Edge("A", "C"))
BACD_graph.add_edge(pydot.Edge("C", "D"))

DCBA_graph = pydot.Dot(graph_type='digraph')
DCBA_graph.add_node(pydot.Node("A"))
DCBA_graph.add_node(pydot.Node("B"))
DCBA_graph.add_node(pydot.Node("C"))
DCBA_graph.add_node(pydot.Node("D"))
DCBA_graph.add_edge(pydot.Edge("A", "B"))
DCBA_graph.add_edge(pydot.Edge("B", "D"))
DCBA_graph.add_edge(pydot.Edge("C", "A"))
DCBA_graph.add_edge(pydot.Edge("C", "D"))

#graph_image(ABCD_graph); graph_image(BACD_graph); graph_image(DCBA_graph)

s4 = random_state(A @ B @ C @ D)

print("\n* Four-node case")
print( s4 )
print("ABCD match: ", state_graph_match(s4, ABCD_graph) )
print("BACD match: ", state_graph_match(s4, BACD_graph) )
print("DCBA match: ", state_graph_match(s4, DCBA_graph) )




# Example from Marco Gaobardi

mg = State([0.01, 0.03, 0.18, 0.06,
            0.04, 0.12, 0.045, 0.015, 
            0.035, 0.105, 0.0675, 0.0225,
            0.015, 0.045, 0.1575, 0.0525], A @ B @ C @ D)

print("\n* Marco's perfect map example")
#print("State check: ", mg >= truth(A @ B @ C @ D) )

mg_graph = pydot.Dot(graph_type='digraph')
mg_graph.add_node(pydot.Node("A"))
mg_graph.add_node(pydot.Node("B"))
mg_graph.add_node(pydot.Node("C"))
mg_graph.add_node(pydot.Node("D"))
mg_graph.add_edge(pydot.Edge("A", "B"))
#mg_graph.add_edge(pydot.Edge("A", "C"))
mg_graph.add_edge(pydot.Edge("C", "B"))
mg_graph.add_edge(pydot.Edge("C", "D"))

graph_image(mg_graph)

print("mg match: ", state_graph_match(mg, mg_graph) )


"""
