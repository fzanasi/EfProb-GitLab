from efprob_dc import *

dom = R(0,20)

g = gaussian_state(10,20,dom)

perf = chan_fromklmap(lambda x: gaussian_state(x,1,R(-1,1)), dom, R(-1,1))

comp = Predicate(lambda x,y: x < y, [R(-1,1), R(-1,1)])

prior = g @ g

prior.plot()

#posterior = prior / ((perf @ perf) << comp)

#posterior.plot()



# import matplotlib.pyplot as plt
# # for 3D plots
# from mpl_toolkits.mplot3d import Axes3D

# def plot2(s, dom=None):
#     steps = 100 # number of computations on the line
#     fig, (ax) = plt.subplots(1, 1, figsize=(10,5))
#     ax = fig.add_subplot(111, projection='3d')
#     if dom is None:
#         Xdom = s.dom[0]
#         Ydom = s.dom[1]
#     X = np.arange(Xdom[0], Xdom[1], (Xdom[1]-Xdom[0])/steps)
#     Y = np.arange(Ydom[0], Ydom[1], (Ydom[1]-Ydom[0])/steps)
#     X, Y = np.meshgrid(X, Y)
#     zs = np.array([s.getvalue(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#     Z = zs.reshape(X.shape)
#     ax.plot_surface(X, Y, Z)
#     plt.draw()
#     #plt.savefig("/tmp/fun2.png", dpi=72)
#     plt.pause(0.001)
#     input("Press [enter] to continue.")
#     return None
