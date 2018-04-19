# Examples based on Barber's book

from efprob_dc import *

print("\nExample 1.2")
print("===========")

print("Hamburger eating; unsolved")

KJ = flip(1/100000, dom = ['k', '~k'])
H = flip(0.5, dom = ['h', '~h'])

r = random.uniform(0,1)

c = chan_from_states([flip(0.9, dom = H.dom), flip(r, dom = H.dom)], KJ.dom)

print("Joint state of KJ and hamburger eating, depending on unknown:")
joint = tuple_channel(c, idn(KJ.dom)) >> KJ
print( joint )
print("We are given that its hamburger-marginal is uniform (0.5); however, so far we get as marginal:")
print( joint % [1,0] )

item_swap = chan_from_states([flip(0, dom = H.dom), flip(1, dom = H.dom)], H.dom)

up_joint = joint / ((item_swap >> (joint % [1,0])).as_pred() @ truth(KJ.dom))

print("After a suitable update we get as marginals:")
print( up_joint % [1,0] )
print( up_joint % [0,1] )
print("We get the same, unknown-dependent, outcome via inversion")
print( c.inversion(KJ) >> H  )

# Analytically we have P(H) = 0.9 * 1/100000 + r * 999999/100000.
# It is given that this value is 0.5. Hence:
# r = (0.5 - 0.9 * 1/100000) * 99999 / 100000


print("\nExample 1.3")
print("===========")

print("Probability that Butler is murdered")

ButlerVictim = bool_dom
MaidVictim = bool_dom
Knife = bool_dom

b = flip(0.6, dom=ButlerVictim)
m = flip(0.2, dom=MaidVictim)

d = chan_from_states([flip(0.1), flip(0.6), flip(0.2), flip(0.3)],
                     [ButlerVictim, MaidVictim])

print( (b @ m) / (d << yes_pred) % [1,0] )


print("\nExample 1.5")
print("===========")

print("Transitivity of implication")

r = random.uniform(0,1)
s = random.uniform(0,1)

c = chan_from_states([flip(1), flip(r)], bool_dom)
d = chan_from_states([flip(1), flip(s)], bool_dom)

print( d >> (c >> (random_state(bool_dom) / yes_pred)) )

print("\nExample 1.6")
print("===========")

print("Inverse Modus ponens; contrapositive")

r = random.uniform(0,1)

c = chan_from_states([flip(1), flip(r)], bool_dom)

print( random_state(bool_dom) / (c >> yes_pred) )


print("\nExample 3.1")
print("===========")

print("Was it the burglar?")

B = flip(0.01)
E = flip(0.000001)
a = chan_from_states([flip(0.9999), flip(0.99), flip(0.99), flip(0.0001)], 
                     [bool_dom, bool_dom])
r = chan_from_states([flip(1), flip(0)], bool_dom)

print("Probability of a burglary, given that the alarm is sounding")
print( (B @ E) / (a << yes_pred) % [1,0] >= yes_pred,
       (B @ E) / (a << yes_pred) % [1,0] >= no_pred )
print("Now if additionally there is a warning on the radio")
print( (B @ (E / (r << yes_pred))) / (a << yes_pred) % [1,0] >= yes_pred,
       (B @ (E / (r << yes_pred))) / (a << yes_pred) % [1,0] >= no_pred )

print("\nExample 3.2")
print("===========")

print("Now with soft/uncertain evidence")

print("I would guess the natural interpretation would be:")
print( (B @ E) / (a << flip(0.7).as_pred()) % [1,0]  )
print("However, the book computes:")
p = Predicate([(B @ E) / (a << yes_pred) % [1,0] >= yes_pred,
               (B @ E) / (a << no_pred) % [1,0] >= yes_pred], bool_dom)
print( flip(0.7) >= p )
print("Induced evidence on B and E")
print( a << flip(0.7).as_pred() )
print( (B @ E) / (a << flip(0.7).as_pred()) )


print("\nSimpson's paradox 3.4.1")
print("=======================")

# Domains for Gender, Drugs, Recovery
G = ['M', 'F']
D = ['d', '~d']
R = ['r', '~r']

table = State([18/80, 12/80, 7/80, 3/80, 
               2/80, 8/80, 9/80, 21/80], [G,D,R])

print("* Overview table:")
print( table )
print("With marginals:")
print( table % [1,0,0] )
print( table % [0,1,0] )
print( table % [0,0,1] )
print("* Recovery rates for males: ",
      table / (point_pred('M',G) @ truth(D) @ truth(R)) % [0,0,1] )
print("Recovery rates for females: ",
      table / (point_pred('F',G) @ truth(D) @ truth(R)) % [0,0,1] )
print("Total recovery rates with/without drugs, as in the (lower right part of the table in the book):")
print(  table / (truth(G) @ point_pred('d',D) @ truth(R)) % [0,0,1] )
print(  table / (truth(G) @ point_pred('~d',D) @ truth(R)) % [0,0,1] )
print("* Recovery rates for males/females with/without drugs, as in the table in the book, but here via disintegration:")
rec = table // [1,1,0]
print( rec('M', 'd') )
print( rec('M', '~d') )
print( rec('F', 'd') )
print( rec('F', '~d') )
print("* Post intervention calculation as in the book, with (uniform) gender marginal inserted:")
print( (rec * ((table % [1,0,0]).as_chan() @ idn(D)))('d') )
print( (rec * ((table % [1,0,0]).as_chan() @ idn(D)))('~d') )
print("For comparison, total recovery rates as computed earlier, now via conditioning:")
print( table[ [0,0,1] : [0,1,0] ]('d') )
print( table[ [0,0,1] : [0,1,0] ]('~d') )


print("\nExercise 3.6")
print("============")

print("Car start problem")

B = Dom(['b', 'g'], names = "Battery")
F = Dom(['e', 'f'], names= "Fuel")
G = Dom(['e', 'f'], names = "Gauge")
T = Dom([True, False], names = "TurnOver")
S = Dom([True, False], names= "Start")

# initial states
iB = State([0.02, 0.98], B)
iF = State([0.05, 0.95], F)

# channels
cG = chan_from_states([State([0.97, 0.03], G), 
                       State([0.04, 0.96], G), 
                       State([0.99, 0.01], G), 
                       State([0.1, 0.9], G)], 
                      B @ F)

cT = chan_from_states([State([0.02, 0.98], T), 
                       State([0.97, 0.03], T)], 
                      B)

cS = chan_from_states([State([0.08, 0.92], S), 
                       State([0.99, 0.01], S), 
                       State([0.01, 0.99], S), 
                       State([0.0, 1.0], S)], 
                      T @ F)

start_joint = ((idn(B) @ idn(T) @ cS @ idn(G) @ idn(F)) \
               * (idn(B) @ copy(T) @ swap(G,F) @ idn(F)) \
               * (idn(B) @ cT @ cG @ idn(F) @ idn(F)) \
               * (copy(B,3) @ copy(F,3))) >> (iB @ iF) 

# no start predicate
ns = point_pred(False, S)

print( start_joint / (truth(B) @ truth(T) @ ns @ truth(G) @ truth(F)) \
       % [0,0,0,0,1])

print(  (iB @ iF) / ((cT @ idn(F)) << (cS << ns)) % [0,1] )
