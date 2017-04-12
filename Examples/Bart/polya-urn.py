from efprob_dc import *

# https://probmods.org/chapters/05-observing-sequences.html

print("\nPolya's Urn\n")

N = 8

#
# pairs of black and with numbers of balls in the urn
#
num_dom = [range(1,N), range(1,N)]
prior = point_state((1,1), num_dom)
col_dom = ['B', 'W']

c = chan_fromklmap(lambda b,w:
                   b/(b+w) * (point_state('B', col_dom) \
                              @ point_state((b+1,w), num_dom)) \
                   +
                   w/(b+w) * (point_state('W', col_dom) \
                              @ point_state((b,w+1), num_dom)),
                   num_dom, [col_dom] + num_dom)

def cn(n):
    if n==1:
        return c
    d = cn(n-1)
    m = len(d.cod) - 2
    return (idn(m*[col_dom]) @ c) * d

def bwn(n):
    return (idn(n*[col_dom]) @ discard(num_dom)) * cn(n)

(bwn(6) >> prior).plot()
