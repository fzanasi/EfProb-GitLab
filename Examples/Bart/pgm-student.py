from pgm_efprob import *
import timeit

print("\nStudent model")
print("=============\n")

# copied from:
# https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb

student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# Defining individual CPDs.
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])

cpd_g = TabularCPD(variable='G', variable_card=3, 
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                   evidence=['I', 'D'],
                   evidence_card=[2, 2])

cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])

student_model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

print("Model correct: ",  student_model.check_model() )

student_inference = VariableElimination(student_model)

student_graph = pydot_graph_of_pgm(student_model)
#graph_image(student_graph, "student")

#print( efprob_domains_of_pgm(student_model) )

student_cpts = efprob_channels_of_pgm(student_model)

#print(student_cpts)

student_stretch = stretch(student_model, observed=True)

student_joint = evaluate_stretch(student_stretch['channels'])

print("\nStudent joint state")

student_joint = reorder_state_domains(student_joint, 
                                      ['D', 'I', 'G', 'L', 'S'])

print("\nStudent joint state")
#print( student_joint )

print("\nMarginals")
print( student_joint % [1,0,0,0,0] )
print( student_inference.query(['D'])['D'] )
print( student_joint % [0,1,0,0,0] )
print( student_inference.query(['I'])['I'] )
print( student_joint % [0,0,1,0,0] )
print( student_cpts['G'] >> (student_cpts['I'] @ student_cpts['D']) )
print( student_inference.query(['G'])['G'] )
print( student_joint % [0,0,0,0,1] )
print( student_inference.query(['S'])['S'] )
print( student_joint % [0,0,0,1,0] )
print( student_inference.query(['L'])['L'] )

print("\nUpdate tests")
print("============\n")

student_dom = student_joint.dom

p = point_pred('S_0', student_dom[4])

print( student_joint / (truth(student_dom[0:4]) @ p) % [0,0,0,1,0] )

print( student_cpts['L'] >> 
       (student_cpts['G'] >> 
        ((student_cpts['I']/(student_cpts['S'] << p)) @ student_cpts['D'])) )

print( student_inference.query(['L'], evidence={'S': 0})['L'] )


N = 100

print("\nInference timing comparison,", N, "times\n")

student_stretch = stretch(student_model)

t1 = timeit.timeit(lambda: student_inference.query(['L'], 
                                                   evidence={'S': 0})['L'], 
                   number=N)



t2 = timeit.timeit(lambda: 
                   inference_query(student_stretch, 'L', {'S' : [0,1]}),
                   number = N) 

t3 = timeit.timeit(lambda: student_joint/(truth(student_dom[0:4]) @ p) % [0,0,0,1,0], 
                   number=N)


print("Times for: variable elimination (t1), transformations (t2), joint-update (t3)")
print(t1)
print(t2)
print(t3)
print("Fractions t1/t2, t1/t3")
print(t1/t2)
print(t1/t3)







