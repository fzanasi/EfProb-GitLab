from efprob_dc import *
import efprob_qu as qu
from math import *
import matplotlib.pyplot as plt
# for 3D plots
from mpl_toolkits.mplot3d import Axes3D

# http://www.mathpages.com/home/kmath521/kmath521.htm

print("\nTomography experiments")

def bloch_plot(s):
    steps = 100 # number of computations on the line
    fig, (ax) = plt.subplots(1, 1, figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    #xs = np.linspace(f.bounds[0][0], f.bounds[0][1], 10, endpoint=True)
    X = np.arange(0, pi, pi/steps)
    Y = np.arange(0, 2*pi, 2*pi/steps)
    X, Y = np.meshgrid(X, Y)
    zs = np.array([s.getvalue(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    plt.draw()
    #plt.savefig("/tmp/fun2.png", dpi=72)
    plt.pause(0.001)
    input("Press [enter] to continue.")
    return None

#
# Domain, with uniform distribution on it
#
bloch_dom = [R(0,pi), R(0,2*pi)]
prior = uniform_state(bloch_dom)


#
# x/y/z channels, mapping coordinates of the domain to coin whose bias
# is given by the validity of the x/y/z predicate in the state
# corresponding to the coordinates
#
x_tom = chan_fromklmap(lambda th, ph: 
                       flip( qu.bloch_state(th, ph) >= qu.x_pred ), 
                       bloch_dom, 
                       bool_dom)

y_tom = chan_fromklmap(lambda th, ph: 
                       flip( qu.bloch_state(th, ph) >= qu.y_pred ), 
                       bloch_dom, 
                       bool_dom)

z_tom = chan_fromklmap(lambda th, ph: 
                       flip( qu.bloch_state(th, ph) >= qu.z_pred ), 
                       bloch_dom, 
                       bool_dom)

#
# Random variables, used for expected coordinate function
#
rv1 = randvar_fromfun(lambda x: x, bloch_dom[0]) @ truth(bloch_dom[1])
rv2 = truth(bloch_dom[0]) @ randvar_fromfun(lambda x: x, bloch_dom[1])
def exp_coord(s): return [rv1.exp(s), rv2.exp(s)]

#
# x/y/z coins in uniform (prior) state
#
print( x_tom >> prior )
print( y_tom >> prior )
print( z_tom >> prior )


#
# Ad hoc posterior state
#
posterior = prior / (z_tom << yes_pred) / (y_tom << no_pred)

#bloch_plot(posterior)

#
# Experimental code for trying to re-learn a known state
#

pred = {
    'x' : x_tom << yes_pred,
    '~x' : x_tom << no_pred,
    'y' : y_tom << yes_pred,
    '~y' : y_tom << no_pred,
    'z' : z_tom << yes_pred,
    '~z' : z_tom << no_pred
}


sample_size = 10
th = random.uniform(0,pi)
ph = random.uniform(0,2*pi)

print("\nArbitrary state, to be learned, with polar coordinates:")
print(th, ph)

rx = qu.bloch_state(th, ph) >= qu.x_pred
ry = qu.bloch_state(th, ph) >= qu.y_pred
rz = qu.bloch_state(th, ph) >= qu.z_pred

print("Probabilities in x/y/z direction: ", rx, ry, rz )

sample_x = np.random.multinomial(sample_size, [rx, 1-rx])
sample_y = np.random.multinomial(sample_size, [rx, 1-rx])
sample_z = np.random.multinomial(sample_size, [rx, 1-rx])

print("Samples for x/y/z: ", sample_x, sample_y, sample_z )


observations = ['x'] * sample_x[0] + ['~x'] * sample_x[1] \
               + ['y'] * sample_y[0] + ['~x'] * sample_y[1] \
               + ['z'] * sample_z[0] + ['~z'] * sample_z[1]


s = prior
for ob in observations:
    s = s / pred[ob]

print("Expected polar coordinates of the learned state: ", exp_coord(s) )
#print( qu.bloch_state(th, ph) )
#print( qu.bloch_state(*exp_coord(s)) )
print("Trace distance between original and learned state: ", 
      qu.trdist(qu.bloch_state(th, ph), qu.bloch_state(*exp_coord(s))) )

bloch_plot(s)








