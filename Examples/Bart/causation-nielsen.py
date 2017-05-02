from efprob_dc import *

# http://www.michaelnielsen.org/ddi/if-correlation-doesnt-imply-causation-then-what-does/
#
# See also:
#
# Cambridge Advanced Tutorial Lecture Series on Machine Learning
# CAUSALITY
# Ricardo Silva
# Gatsby Computational Neuroscience Unit
# rbas@gatsby.ucl.ac.uk
# http://mlg.eng.cam.ac.uk/zoubin/tut06/cambridge_causality.pdf

print("\nSimpson's paradox")

# Democrates and republicans
Dem = range(248)
Rep = range(172)

uDem = uniform_state(Dem)
uRep = uniform_state(Rep)

# Predicate: from the North
NDem = pred_fromfun(lambda x: x < 154, Dem)
NRep = pred_fromfun(lambda x: x < 162, Rep)

print( uDem >= NDem, uRep >= NRep )

DemCRA = pred_fromfun(lambda x: x < 145 or (154 <= x and x < 161), Dem)

print("\nDemocrates, in North, South, total")
print( uDem / NDem >= DemCRA, uDem / ~NDem >= DemCRA, uDem >= DemCRA )

RepCRA = pred_fromfun(lambda x: x < 138, Rep)

print("\nRepublicans, in North, South, total")
print( uRep / NRep >= RepCRA, uRep / ~NRep >= RepCRA, uRep >= RepCRA )


print("\nKidney stone treatment")

# Data copied from:
# https://en.wikipedia.org/wiki/Simpson%27s_paradox#Kidney_stone_treatment

dom = range(350)
u = uniform_state(dom)

SmallA = pred_fromfun(lambda x: x < 87, dom)
SmallB = pred_fromfun(lambda x: x < 270, dom)

SuccessA = pred_fromfun(lambda x: x < 81 or (87 <= x and x < 87 + 192), dom)
SuccessB = pred_fromfun(lambda x: x < 234 or (270 <= x and x < 270 + 55), dom)

print("Treatmeant A, for small, large, total")
print( u / SmallA >= SuccessA, u / ~SmallA >= SuccessA, u >= SuccessA )
print("Treatmeant B, for small, large, total")
print( u / SmallB >= SuccessB, u / ~SmallB >= SuccessB, u >= SuccessB )



print("\nSmokers with tar and cancer")

def do(w, x):
    return (w % [1,0]) @ (w.disintegration([1,0])(x))

smoke_dom = ['S', '~S']
tar_dom = ['T', '~T']

#  p(\mbox{cancer} | \mbox{do}(smoking)) = 45.25 
# via:
#  p(y| \mbox{do}(x)) = \sum_{x'z} p(y|x',z) p(z|x) p(x') . 
#
# \sum_{x',z) p(cancer|x',z) p(z|smoking) p(x')

# orginal, with lots of symmetry:
st = State([0.475, 0.025, 0.025, 0.475], [smoke_dom, tar_dom])
# own, adapted, non-symmetric version:
#st = State([0.485, 0.025, 0.015, 0.475], [smoke_dom, tar_dom])

print("Smoke marginal: ", st % [1,0])

c = chan_from_states([flip(0.85), flip(0.9), flip(0.05), flip(0.1)],
                     [smoke_dom, tar_dom])

print("Get cancer: ", c >> st >= yes_pred )
print("Smoking, given cancer: ", 
      c >> (st / (point_pred('S', smoke_dom) @ truth(tar_dom))) >= yes_pred )

print( (graph(c) >> st).disintegration([1,0,0])('S') % [0,1] >= yes_pred )

sc = ((graph(c) >> st) % [1,0,1])

print( st % [1,0] )

# p(cancer|smoking,tar) * p(tar|smoking) * p(smoking) + 
# p(cancer|smoking,~tar) * p(~tar|smoking) * p(smoking) +
# p(cancer|~smoking,tar) * p(tar|smoking) * p(~smoking) +
# p(cancer|~smoking,~tar) * p(~tar|smoking) * p(~smoking) 

print( st )
print("Do: ", do(st, 'S') )
print( c >> do(st, 'S') )
print("Don't: ", do(st, '~S') )
print( c >> do(st, '~S') )

print("Do intervention: ",
      c >> (st % [1,0] @ st.disintegration([1,0])('S')) >= yes_pred )

#print( 0.85 * 0.95 * 0.5 + 0.9 * 0.05 * 0.5 + 0.05 * 0.95 * 0.5 + 0.1 * 0.05 * 0.5 )

#print( 0.85 * 0.95 * 0.51 + 0.9 * 0.05 * 0.51 + 0.05 * 0.95 * 0.49 + 0.1 * 0.05 * 0.49 )

print( c * (idn(smoke_dom) @ (st // [1,0])) * copy(smoke_dom) >> point_state('S', smoke_dom) )
