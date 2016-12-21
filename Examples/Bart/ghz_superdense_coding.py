from quantprob import *

# Sources: J. L. Cereceda (2001): Quantum Dense coding using three
# qubits. At http://arxiv.org/abs/quant-ph/0105096.
#
# Anne Hillebrand, Superdense Coding with GHZ and Quantum Key
# Distribution with W in the ZX -calculus, https://arxiv.org/1210.0650

# An additional channel

iy_matrix = np.array([[0,1],
                      [-1,0]])

iy_chan = channel_from_unitary(iy_matrix, Dom([2]))

# The classical information that will be transferred

r1 = random_probabilistic_state(2)
r2 = random_probabilistic_state(2)
r3 = random_probabilistic_state(2)

# combined into one state of type Dom([8])

r = kron(4,2) >> ((kron(2,2) >> (r1 @ r2)) @ r3)

print(r1)
print(r2)
print(r3)

alice = (discard(8) @ idn(2,2,2)) \
        >> ((ccase(idn(2) @ idn(2), 
                   idn(2) @ x_chan, 
                   x_chan @ idn(2), 
                   x_chan @ x_chan, 
                   z_chan @ idn(2), 
                   z_chan @ x_chan, 
                   iy_chan @ idn(2), 
                   iy_chan @ x_chan) @ idn(2)) \
            >> (r @ ghz))


bob = (kroninv(2,2) @ idn(2)) * kroninv(4,2) * meas_ghz 

ghz_superdense_coding = bob >> alice

# The classical input re-appears, but in reverse order

print( ghz_superdense_coding % [1,0,0] )
print( ghz_superdense_coding % [0,1,0] )
print( ghz_superdense_coding % [0,0,1] )


