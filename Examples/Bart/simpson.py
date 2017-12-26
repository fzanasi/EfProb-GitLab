from efprob_dc import *

# from: Dawid, FUNDAMENTALS OF STATISTICAL CAUSALITY

# total number 800

#
# Domains for Gender (M/F), Treatment (Y/N), Result (R/D) 
#    where R = Recovered, D = Died
#
gd = Dom(['M', 'F'])
td = Dom(['Y', 'N'])
rd = Dom(['R', 'D'])

dom = gd @ td @ rd

# typo corrected: 220 -> 210

s = State([180/800, 120/800, 70/800, 30/800,
           20/800, 80/800, 90/800, 210/800], dom)

print( s )

# Gender --> Treatment x Result
g2tr = s[ [0,1,1] : [1,0,0] ]

#print("\nMale results")
#print( g2tr('M') )
#print( g2tr('M') % [0,1] )

# Gender x Treatment --> Result
gt2r = s[ [0,0,1] : [1,1,0] ]

print("\nMale Y/N Treatment")
print( gt2r('M', 'Y') )
print( gt2r('M', 'N') )

print("\nFemale Y/N Treatment")
print( gt2r('F', 'Y') )
print( gt2r('F', 'N') )

# Treatment --> Result
t2r = s[ [0,0,1] : [0,1,0] ]
print("\nY/N Treatment")
print( t2r('Y') )
print( t2r('N') )
