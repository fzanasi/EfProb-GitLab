#
# Borgstrom et al.
# https://arxiv.org/abs/1308.0689
#
# Naive Bayesian Classifier, Single Feature Training, section 3
#
from efprob_dc import *

g = chan_fromklmap(lambda m,s: gaussian_state(m,s), [R(0,10), R(0,1)], R)

w0 = uniform_state([R(0,10), R(0,1)])


# [6, 5.92, 5.58, 5.92]

w1 = g.inversion(w0)(6)

w2 = g.inversion(w1)(5.92)

w3 = g.inversion(w2)(5.58)

w4 = g.inversion(w3)(5.92)

# print( w4 )

# 5.855

#print( (w4 % [1,0]).expectation() )

# 3.5033e-02 ??

#print( (w4 % [0,1]).expectation() )

# prints
# 5.85500000009
# 0.319375718814




"""

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
classifier = chan_from_states([gaussian_state(5.855, math.sqrt(3.5033e-02)) @
                               gaussian_state(176.25, math.sqrt(1.2292e+02)) @
                               gaussian_state(11.25, math.sqrt(9.1667e-01)),
                               gaussian_state(5.4175, math.sqrt(9.7225e-02)) @
                               gaussian_state(132.5, math.sqrt(5.5833e+02)) @
                               gaussian_state(7.5, math.sqrt(1.6667e+00))], Sex)


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


"""

print("\nWheather-Playing")

# From: https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/

#
# Weather options: Sunny, Overcast, Rainy
#
Weather = ['S', 'O', 'R']

table = State([3/14, 2/14, 4/14, 0, 2/14, 3/14], [Weather, bool_dom])

print("Table, as joint distribution:")
print(table)

w2p = table.disintegration([1,0])

print("\nProbabilities of playing in different kinds of weather")
print("Sunny: ", w2p('S') )
print("Overcast: ", w2p('O') )
print("Rain: ", w2p('R') )

p2w = table.disintegration([0,1])

print("\nWeather probabilities depending on playing:")
print("Playing: ", p2w(True))
print("Not playing: ", p2w(False))


print("\nFruit classification")

# From: http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification
#
# Also: http://blog.aylien.com/naive-bayes-for-dummies-a-simple-explanation/


#
# Fruit: Banana (B), Orange (O), Other (N)
#
Fruit = ['B', 'O', 'N']

classifier = chan_from_states([flip(400/500) @ flip(350/500) @ flip(450/500),
                               flip(0/300) @ flip(150/300) @ flip(300/300),
                               flip(100/200) @ flip(150/200) @ flip(50/200)],
                              Fruit)

fruits = State([0.5, 0.3, 0.2], Fruit)

print( classifier.inversion(fruits)(True,True,True) )
print("This corresponds to the given solution, because the following two ratios are equal")
print(0.252 / 0.01875, 0.931/ 0.0693 )


print("\nRam Narasimhan example")

# p.351 of 

Age = ['<30', '30-40', '>40']
Income = ['H', 'M', 'L']
Student = ['S', '~S']
Credit = ['F', 'E']
Buys = ['C', '~C']

dom = [Age, Income, Student, Credit]

# Table is given with the following 14 entries

# <=30,low,yes,fair,yes
# <=30,medium,yes,excellent,yes
# 31-40,high,no,fair,yes
# 31-40,low,yes,excellent,yes
# 31-40,medium,no,excellent,yes
# 31-40,high,yes,fair,yes
# >40,medium,no,fair,yes
# >40,low,yes,fair,yes
# >40,medium,yes,fair,yes

# <=30,high,no,fair,no
# <=30,high,no,excellent,no
# <=30,medium,no,fair,no
# >40,medium,no,excellent,no
# >40,low,yes,excellent,no

buys_prior = State([9/14, 5/14], Buys)

print("prior: ", buys_prior )

classifier = chan_from_states(
    [State([2/9, 4/9, 3/9], Age) @
     State([2/9, 4/9, 3/9], Income) @
     State([6/9, 3/9], Student) @
     State([6/9, 3/9], Credit),
     State([3/5, 0, 2/5], Age) @
     State([2/5, 2/5, 1/5], Income) @
     State([1/5, 4/5], Student) @
     State([2/5, 3/5], Credit)], Buys)

print("\nThe next conditional probabilities differ from the ones on the web")
print("\nConditionals for not-buy")
print( classifier('~C') % [1,0,0,0] )
print( classifier('~C') % [0,1,0,0] )
print( classifier('~C') % [0,0,1,0] )
print( classifier('~C') % [0,0,0,1] )
print("\nConditionals for buy")
print( classifier('C') % [1,0,0,0] )
print( classifier('C') % [0,1,0,0] )
print( classifier('C') % [0,0,1,0] )
print( classifier('C') % [0,0,0,1] )

print("\nBuys: ", classifier.inversion(buys_prior)('<30', 'M', 'S', 'F') )
     
# solution at stackoverflow:
# yes  ==>  0.0720164609053
# no  ==>  0.0411428571429

print( 0.0720164609053 / 0.0411428571429, 0.805 / 0.195 )


print("\nOther Weather-Play predictions")

# https://cse.sc.edu/~rose/587/PPT/NaiveBayes.ppt
#
# Data_Mining_Practical_Machine_Learning_Techniques_and_Tools.pdf

Weather = ['S', 'O', 'R']
dTemp = ['H', 'M', 'C']

# Discrete version:

prior_play = flip(9/14)

dclass = chan_from_states(
    [State([2/9, 4/9, 3/9], Weather) @
     State([2/9, 4/9, 3/9], dTemp) @
     flip(3/9) @
     flip(3/9),
     State([3/5, 0/5, 2/5], Weather) @
     State([2/5, 2/5, 1/5], dTemp) @
     flip(4/5) @
     flip(3/5)], bool_dom)

print("\nDiscrete case:")
print( dclass.inversion(prior_play)('S', 'C', True, True) )

print("Correspondence: ", 0.205 / 0.795, 0.0053 / 0.0206 )

cclass = chan_from_states(
    [State([2/9, 4/9, 3/9], Weather) @
     gaussian_state(73, 6.2) @
     gaussian_state(79.1, 10.2) @
     flip(3/9),
     State([3/5, 0/5, 2/5], Weather) @
     gaussian_state(74.6, 7.9) @
     gaussian_state(86.2, 9.7) @
     flip(3/5)], bool_dom)

print("\nContinuous case:")
# value 88.9 is adjusted, to get the right outcome
# See p.96 of Data_Mining_Practical_Machine_Learning_Techniques_and_Tools.pdf
print( cclass.inversion(prior_play)('S', 66, 90, True) )

print( 0.000036 / (0.000136 + 0.000036), 0.000136 / (0.000136 + 0.000036) )
     

