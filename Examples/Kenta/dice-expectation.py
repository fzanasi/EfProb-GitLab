# Examples from:
# http://www.futureaccountant.com/theory-of-expectation-random-variable/problems-solutions/throwing-rolling-dice.php

from efprob_dc import *

pips = [1,2,3,4,5,6]
s_dice = uniform_state(pips)
rv = randvar(lambda x: x, pips)

print("Expected value: ", rv.exp(s_dice))
print("Variance: ", rv.var(s_dice))
print("Standard deviation: ", rv.stdev(s_dice))

s_2dice = s_dice @ s_dice
rv_sum2 = randvar(lambda x, y: x + y, [pips, pips])

print("Expected value:", rv_sum2.exp(s_2dice))
print("Variance:", rv_sum2.var(s_2dice))
print("Standard deviation:", rv_sum2.stdev(s_2dice))

def rv_sum(n):
    return randvar(lambda *xs: sum(xs), [pips] * n)

for n in [2, 3, 5]:
    print("n = {}:".format(n), rv_sum(n).exp(s_dice ** n))

def rv_even_game(n):
    return randvar(lambda *xs:
                   sum(x if x % 2 == 0 else -x for x in xs),
                   [pips] * n)

for n in [2, 3, 5]:
    print("n = {}:".format(n), rv_even_game(n).exp(s_dice ** n))

rv_prime_game = randvar(lambda x:
                        x if x in [2, 3, 5] else -x,
                        [pips])

print(rv_prime_game.exp(s_dice))

rv_umbrella_sales = randvar(lambda x: 300 if x else -60, [True, False])
s_rain = flip(0.3)
print(rv_umbrella_sales.exp(s_rain))
