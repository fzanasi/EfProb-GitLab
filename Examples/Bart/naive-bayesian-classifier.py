#
# Borgstrom et al.
# https://arxiv.org/abs/1308.0689
#
# Naive Bayesian Classifier, Single Feature Training, section 3
#
from efprob_dc import *

prior = gaussian_state(0.5,1.0, R(0,1))

#prior.plot()

chan = chan_fromklmap(lambda x: gaussian_state(x,1, R(-1,2)), R(0,1), R(-1,2))

#
# Since there is no continuous copy, we have to define the joint
# distribution by hand.
#
joint = State(lambda x,y: prior.getvalue(x) * chan(x).getvalue(y), 
              [R(0,1), R(-1,2)])

dis = joint.disintegration([0,1])

ndis = chan.inversion(prior)

print( ndis )

#dis(1).plot()
#ndis(1).plot()


#
# s prior state, x in [-1,2]
#
def learn(s,x):
    joint = State(lambda x,y: s.getvalue(x) * chan(x).getvalue(y), 
                  [R(0,1), R(-1,2)])
    dis = joint.disintegration([0,1])
    return dis(x)

# Glass
g0 = prior
g1 = learn(g0, 0.18)
#g1.plot()
g2 = learn(g1, 0.21)
#g2.plot()
print( g2.expectation() )

# Watch
w0 = prior
w1 = learn(w0, 0.11)
#w1.plot()
w2 = learn(w1, 0.073)
#w2.plot()
print( w2.expectation() )

# Plate
p0 = prior
p1 = learn(p0, 0.23)
#p1.plot()
p2 = learn(p1, 0.45)
#p2.plot()
print( p2.expectation() )


print("\nSex classification from Wikipedia\n")

# https://en.wikipedia.org/wiki/Naive_Bayes_classifier

Sex = ['M', 'F']

r4 = range(4)
u4 = uniform_state(r4)

MHeight = RandVar([6, 5.92, 5.58, 5.92], r4)
MWeight = RandVar([180, 190, 170, 165], r4)
MFoot = RandVar([12, 11, 12, 10], r4)

print("Male")
print( u4.expectation(MHeight) )
print( u4.variance(MHeight) )
print( u4.expectation(MWeight) )
print( u4.variance(MWeight) )
print( u4.expectation(MFoot) )
print( u4.variance(MFoot) )

FHeight = RandVar([5, 5.5, 5.42, 5.75], r4)
FWeight = RandVar([100, 150, 130, 150], r4)
FFoot = RandVar([6, 8, 7, 9], r4)

print("\nFemale")
print( u4.expectation(FHeight) )
print( u4.variance(FHeight) )
print( u4.expectation(FWeight) )
print( u4.variance(FWeight) )
print( u4.expectation(FFoot) )
print( u4.variance(FFoot) )


classifier = chan_from_states([gaussian_state(u4.expectation(MHeight),
                                              u4.variance(MHeight)) @
                               gaussian_state(u4.expectation(MWeight),
                                              u4.variance(MWeight)) @
                               gaussian_state(u4.expectation(MFoot),
                                              u4.variance(MFoot)),
                               gaussian_state(u4.expectation(FHeight),
                                              u4.variance(FHeight)) @
                               gaussian_state(u4.expectation(FWeight),
                                              u4.variance(FWeight)) @
                               gaussian_state(u4.expectation(FFoot),
                                              u4.variance(FFoot))], Sex)

#
# Numbers copied from Wikipedia; variances differ...
#
classifier = chan_from_states([gaussian_state(5.855, 3.5033e-02) @
                               gaussian_state(176.25, 1.2292e+02) @
                               gaussian_state(11.25, 9.1667e-01),
                               gaussian_state(5.4175, 9.7225e-02) @
                               gaussian_state(132.5, 5.5833e+02) @
                               gaussian_state(7.5, 1.6667e+00)], Sex)


joint = ((idn(Sex) @ classifier) * copy(Sex)) >> uniform_state(Sex)

dis = joint.disintegration([0,1,1,1])

print( classifier )

ndis = classifier.inversion(uniform_state(Sex))

print("")
#
# Returns high male probability, whereas Wikipedia says female. Bug?
#
print( dis(6,130,8) )
print( ndis(6,130,8) )



