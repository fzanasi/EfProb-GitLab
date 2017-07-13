from efprob_qu import *

def cup_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [1]
    for i in range(n-1):
        ls = ls + n*[0] + [1]
    v = np.array(ls)
    mat = np.zeros((n*n,n*n,1,1))
    mat[...,0,0] = np.outer(v.transpose(), v)
    return Channel(1/n * mat, [], dom + dom)

def cup_state(dom):
    return cup_chan(dom) >> init_state

D = Dom([2])

alice = (meas0 @ meas0) * (hadamard @ idn([2])) * cnot

bob = (discard([2]) @ idn([2])) \
      * ccontrol(z_chan) \
      * (idn([2]) @ discard([2]) @ idn([2])) \
      * (idn([2]) @ ccontrol(x_chan))

teleportation = bob * (alice @ idn([2])) * (idn([2]) @ bell00.as_chan())

# s = random_state(D)
# print( s )
# print( teleportation >> s )

#print( (discard(D) @ discard(D)) * alice * (idn(D) @ uniform_probabilistic_state(D).as_chan()) )

w = random_state(D+D)
#w = random_probabilistic_state(D+D)
#w = random_state(D) @ random_state(D)

# force first component to be uniform
#w = convex_state_sum((0.5, (meas0 @ idn((w % [0,1]).dom)) >> w), (0.5, (meas1 @ idn((w % [0,1]).dom)) >> w))

# force second component to be uniform
#w = convex_state_sum((0.5, (idn((w % [1,0]).dom) @ meas0) >> w), (0.5, (idn((w % [1,0]).dom) @ meas1) >> w))

# force first component to be probabilistic
w = (meas0 @ idn((w % [0,1]).dom)) >> w


print(w % [1,0])
print(w % [0,1])


p = random_pred(D)
#p = random_probabilistic_pred(D)
q = random_pred(D)
#q = random_probabilistic_pred(D)

def extract(stat):
    return (idn(D) @ alice) * (stat.as_chan() @ idn(D))

print("\nSanity checks")
print( extract(w) << truth(D) @ truth(D) @ truth(D) )
print( w == (idn(D) @ bob) * (extract(w) @ idn(D)) >> cup_state(D) )

print("\nFirst validity check: ", 
      w >= p @ truth(D),
      w % [0,1] >= extract(w) << (p @ truth(D) @ truth(D)) )
print("Second validity check: ", 
      w >= truth(D) @ q,
      w % [1,0] >= extract(swap >> w) << (q @ truth(D) @ truth(D)) )

# The quotient checks do not work for probabilistic states/predicates
# But they do work if w is a product state
print("\nFirst coordinate conditioning checks: ",
      (w / (p @ truth(D))) % [0,1] == 
      w % [0,1] / (extract(w) << (p @ truth(D) @ truth(D))),
      (w / (p @ truth(D))) % [0,1] == 
      (extract(swap >> w) >> (w % [1,0] / p)) % [1,0,0],
      w % [0,1] / (extract(w) << (p @ truth(D) @ truth(D))) ==
      (extract(swap >> w) >> (w % [1,0] / p)) % [1,0,0] )
      
# Works if first marginal of w is uniform
print( (w / (p @ truth(D))) % [0,1] )
print( w % [0,1] / (extract(w) << (p @ truth(D) @ truth(D))) )
print( (extract(swap >> w) >> (w % [1,0] / p)) % [1,0,0] )


print("\nSecond coordinate conditioning checks: ",
      (w / (truth(D) @ q)) % [1,0] ==
      (extract(w) >> (w % [0,1] / q)) % [1,0,0],
      (w / (truth(D) @ q)) % [1,0] ==
      w % [1,0] / (extract(swap >> w) << (q @ truth(D) @ truth(D))),
      (extract(w) >> (w % [0,1] / q)) % [1,0,0] ==
      w % [1,0] / (extract(swap >> w) << (q @ truth(D) @ truth(D))) )

# Works if second marginal of w is uniform
print( (w / (truth(D) @ q)) % [1,0] )
print( (extract(w) >> (w % [0,1] / q)) % [1,0,0] )
print( w % [1,0] / (extract(swap >> w) << (q @ truth(D) @ truth(D))) )

print("\nExtracted states")
print( (extract(w) >> ket(0)) % [1,0,0] )
print( (extract(w) >> ket(1)) % [1,0,0] )
