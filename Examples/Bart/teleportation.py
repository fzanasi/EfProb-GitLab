from quantprob import *

alice = (meas0 @ meas0) \
        * (hadamard @ idn(2)) \
        * cnot

bob = (discard(2) @ idn(2)) \
      * ccontrol(z_chan) \
      * (idn(2) @ discard(2) @ idn(2)) \
      * (idn(2) @ ccontrol(x_chan)) 

teleportation = bob * (alice @ idn(2)) * (idn(2) @ bell00.as_chan())

print("\n========================")
print("Teleportation as channel")
print("========================")

s = random_state(2)

print("\nRandom input state:\n")

print(s)

print("\nTeleporting it yields:\n")

print(teleportation >> s)

print("\nEquality of channels:\n")

print( teleportation == idn(2) )


# #
# # For gate teleportation, see e.g.:
# #
# # https://www.perimeterinstitute.ca/personal/dgottesman/teleportgates.html
# #

print("\nVariation: gate teleportation")
print("=============================")

def bob_gate(c):
    return proj2 \
        * (idn(2) @ c) \
        * ccontrol(z_chan) \
        * (idn(2) @ proj2) \
        * (idn(2) @ ccontrol(x_chan)) \
        * (idn(2, 2) @ c) \

        
def teleportation_gate(c):
    return bob_gate(c) \
        * (alice @ idn(2)) \
        * (idn(2, 2) @ c) \
        * (idn(2) @ bell00.as_chan())

#ch = hadamard
#ch = x_chan
#ch = y_chan
ch = z_chan

print("\nWhen the gate/channel work is done on Alice's side\n")
print( teleportation_gate(ch) >> s )

print("\nWhen the work is done on Bob's side\n")
print(ch >> (teleportation >> s))

