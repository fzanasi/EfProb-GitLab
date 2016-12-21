from quantprob import *

alice = (discard(2) @ idn(2)) \
        * ccontrol(x_chan) \
        * (idn(2) @ discard(2) @ idn(2)) \
        * (idn(2) @ ccontrol(z_chan)) \
        * (swap @ idn(2))

bob = (meas0 @ meas0) \
      * (hadamard @ idn(2)) \
      * cnot

def superdense_coding_fun(r, s):
    return bob >> ((alice @ idn(2)) >> (cflip(r) @ cflip(s) @ bell00))


print("\n=================")
print("Superdense coding")
print("=================")

print("\nWe pick two random numbers from the unit interval [0,1], namely:\n")

r = random.uniform(0,1)
s = random.uniform(0,1)

print("* ", r)
print("\n* ", s)

print("\nAfter pumping them as probabilistic bits through the superdense coding")
print("channel we get them back as probabilistic states:\n")

sdc = superdense_coding_fun(r,s)

print("* ",  sdc % [1,0] )
print("\n* ",  sdc % [0,1] )


# print("\nChannel version")
# print("===============\n")

# superdense_coding_chan = bob * (alice @ idn(2)) * (classic(2,2) @ bell00.as_chan())

# u = random_probabilistic_state(2)
# v = random_probabilistic_state(2)

# print("Input probabilistic states\n")

# print(u)
# print(v)

# print("\nOutput states\n")

# print( (superdense_coding_chan >> (u @ v)) % [1,0] )

# print( (superdense_coding_chan >> (u @ v)) % [0,1] )

