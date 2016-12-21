from quantprob import *

ghz_teleportation = \
    ( discard(8) @ idn(2) ) \
    * ccase(idn(2), z_chan, x_chan, x_chan * z_chan, \
            z_chan, idn(2), x_chan * z_chan, x_chan) \
    * ( kron(4, 2) @ idn(2) ) \
    * ( meas_bell @ meas_hadamard @ idn(2) ) \
    * (idn(2) @ ghz.as_chan() )

s = random_state(2)

print(s)

print( ghz_teleportation >> s )

print( ghz_teleportation == idn(2) )
