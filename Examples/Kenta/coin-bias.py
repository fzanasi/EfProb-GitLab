from efprob_dc import *

st = uniform_state(R(0, 1))

obs = chan_fromklmap(lambda r: flip(r),
                     R(0, 1), [True, False])
## same as
# obs = channel([lambda xs, _: xs[0],
#                lambda xs, _: 1.0 - xs[0]],
#               R(0, 1), [True, False])


rv = randvar(lambda x: x, R(0, 1))

pred = predicate([1, 0], [True, False])

obs_list = [1,1,1,0,0,0,1,0,1,1,1,1]

for o in obs_list:
    print(rv.exp(st))
    if o:
        st = st / (obs << pred)
    else:
        st = st / (obs << ~pred)

print(rv.exp(st))
st.plot()
