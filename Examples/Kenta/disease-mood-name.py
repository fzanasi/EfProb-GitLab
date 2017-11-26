from efprob_dc import *

print("\nDisease-mood example")
print("====================\n")

bool_list = [True, False]
mood_name = Name("M")
disease_name = Name("D")
mood_dom = Dom(bool_list, names=mood_name)
disease_dom = Dom(bool_list, names=disease_name)
w = State([0.05, 0.4, 0.5, 0.05], mood_dom+disease_dom)
print("Prior joint state")
print( w )
print("with domains and names")
print( w.dom )
print( w.dom.names )
print("with marginals")
w1 = w % [1,0]
w2 = w % [0,1]
print( w1 )
print( w2 )

print("\nA priori chance of positive test")
sens = chan_from_states([flip(9/10), flip(1/20)], disease_dom)
print( sens >> w2 )
print("\nA posteriori change of positive test")
pos_test = sens << yes_pred
print( pos_test )
s = w / (truth(mood_dom) @ pos_test)
print( s % [1,0] )

print("\nDisintegration in the first coordinate")
c1 = w // [1,0]
print( w1 / (c1 << pos_test) )


print("\nDisintegration in the second coordinate")
c2 = w // [0,1]
print( c2 >> (w2 / pos_test) )
