from efprob_dc import *
#
# Theorem: w is non-entwined iff there is no crossover influence
#
# The proof works via point predicates.
#
# Here: nvestigation of conditional independence, commonly expressed as:
#
# p(x,y|z) = p(x|z) p(y|z)
#
# Here it is called conditional non-entwinedness, and expressed via
# disintegration:
#
# s // [1,1,0]  equals the tuple of channels
#
# s % [1,0,1] // [0,1]    s % [0,1,1] // [0,1], 

X = Dom(range(2))
Y = Dom(range(3))
Z = Dom(range(4))

sXZ = random_state(X @ Z)
cZX = sXZ // [0,1]
sYZ = random_state(Y @ Z)
cZY = sYZ // [0,1]
sZ = random_state(Z)

s = (cZX @ cZY @ idn(Z)) * copy(Z,3) >> sZ
print("\n", s )

cZXY = s // [0,0,1]
print("\n", (cZXY @ idn(Z)) * copy(Z) >> sZ )

dZX = s % [1,0,1] // [0,1]
dZY = s % [0,1,1] // [0,1]


#
# The pair dZX @ dZY * copy(Z)  equals  cZXY
#
t = (dZX @ dZY * copy(Z)) @ idn(Z) * copy(Z) >> sZ

print("\n", t )

# p(x|z) = p(x|y,z)

cYZX = s // [0,1,1]

print("\nIndependence of X,Y, given Z")
print( dZX(0) )
print( cYZX(0,0) )
print( cYZX(1,0) )
print( cYZX(2,0) )
print( dZX(1) )
print( cYZX(0,1) )
print( cYZX(1,1) )
print( cYZX(2,1) )
print( dZX(2) )
print( cYZX(0,2) )
print( cYZX(1,2) )
print( cYZX(2,2) )
print( dZX(3) )
print( cYZX(0,3) )
print( cYZX(1,3) )
print( cYZX(2,3) )

pX = random_pred(X)
pY = random_pred(Y)
pZ = point_pred(3, Z)

print("\nConditioning, with point-predicate on Z")
print( s / (truth(X) @ pY @ pZ) % [1,0,0] )
print( s / (truth(X) @ truth(Y) @ pZ) % [1,0,0] )
print( s / (pX @ truth(Y) @ pZ) % [0,1,0] )
print( s / (truth(X) @ truth(Y) @ pZ) % [0,1,0] )

print("\nAdditional parameter W")

# Redefine letters to be consistent with Pearl

Y = Dom(range(2))
Z = Dom(range(3))
X = Dom(range(4))
W = Dom(range(5))


#
# Create state t on X @ Y @ Z @ W 
# with X,Y conditionally non-entwined, given Z,W
#

tYXW = random_state(Y @ X @ W)
dYXW = tYXW // [0,1,1]
tZXW = random_state(Z @ X @ W)
dZXW = tZXW // [0,1,1]
tXW = random_state(X @ W)

t = (dYXW @ dZXW @ idn(X @ W)) \
    * (idn(X) @ idn(W) @ idn(X) @ swap(X,W) @ idn(W)) \
    * (idn(X) @ idn(W) @ copy(X) @ copy(W)) \
    * (idn(X) @ swap(X,W) @ idn(W)) \
    * (copy(X) @ copy(W)) \
    >> tXW

print( t )
