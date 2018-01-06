from pgm_efprob import *
import timeit


print("\nVisit to Asia")
print("=============\n")

edges_list = [('VisitToAsia', 'Tuberculosis'),
              ('LungCancer', 'TuberculosisOrCancer'),
              ('Smoker', 'LungCancer'),
              ('Smoker', 'Bronchitis'),
              ('Tuberculosis', 'TuberculosisOrCancer'),
              ('Bronchitis', 'Dyspnea'),
              ('TuberculosisOrCancer', 'Dyspnea'),
              ('TuberculosisOrCancer', 'Xray')]

cpd_smoker = TabularCPD(variable='Smoker', variable_card=2,
                        values=[[0.5], [0.5]])

cpd_visittoasia = TabularCPD(variable='VisitToAsia', variable_card=2,
                             values=[[0.01], [0.99]])

cpd_lungcancer = TabularCPD(variable='LungCancer', variable_card=2, 
                            values=[[0.1, 0.01], [0.9, 0.99]],
                            evidence=['Smoker'], evidence_card=[2])

cpd_tuberculosis = TabularCPD(variable='Tuberculosis', variable_card=2, 
                              values=[[0.05, 0.01], [0.95, 0.99]],
                              evidence=['VisitToAsia'], evidence_card=[2])

cpd_bronchitis = TabularCPD(variable='Bronchitis', variable_card=2, 
                            values=[[0.6, 0.3], [0.4, 0.7]],
                            evidence=['Smoker'], evidence_card=[2])

cpd_tuberculosisorcancer = TabularCPD(variable='TuberculosisOrCancer', variable_card=2, 
                    # [[1, 0, 0, 1], [0, 1, 0, 1]]
                   values=[[1, 0, 0, 0],
                           [0, 1, 1, 1]],
                   evidence=['LungCancer', 'Tuberculosis'],
                   evidence_card=[2, 2])

cpd_dyspnea = TabularCPD(variable='Dyspnea', variable_card=2, 
                   values=[[0.9, 0.7, 0.8, 0.1],
                           [0.1, 0.3, 0.2, 0.9]],
                   evidence=['Bronchitis', 'TuberculosisOrCancer'],
                   evidence_card=[2, 2])

cpd_xray = TabularCPD(variable='Xray', variable_card=2, 
                      values=[[0.98, 0.05], [0.02, 0.95]],
                      evidence=['TuberculosisOrCancer'], evidence_card=[2])

asia_model = BayesianModel(edges_list)

asia_model.add_cpds(cpd_smoker, cpd_visittoasia, 
                    cpd_lungcancer, cpd_tuberculosis,
                    cpd_bronchitis, cpd_tuberculosisorcancer,
                    cpd_dyspnea, cpd_xray)

print("Asia correct: ", asia_model.check_model() )


asia_graph = pydot_graph_of_pgm(asia_model)
#graph_image(asia_graph, "asia")

# For timing of stretching:
#print(timeit.timeit(lambda: stretch(asia_model, graph_output=False), 
#                    number=1))

asia_inference = VariableElimination(asia_model)

#asia_cpts = efprob_channels_of_pgm(asia_model)

# print( asia_cpts['Smoker'] )
# print( asia_cpts['VisitToAsia'] )
# print( asia_cpts['LungCancer']('Smoker_0') )
# print( asia_cpts['LungCancer']('Smoker_1') )
# #print( asia_cpts['Tuberculosis'] )
# print( asia_cpts['Tuberculosis']('VisitToAsia_0') )
# print( asia_cpts['Tuberculosis']('VisitToAsia_1') )
# #print( asia_cpts['Bronchitis'] )
# print( asia_cpts['Bronchitis']('Smoker_0') )
# print( asia_cpts['Bronchitis']('Smoker_1') )
# #print( asia_cpts['TuberculosisOrCancer'] )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_0') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_1') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_0') )
# print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_1') )
# #print( asia_cpts['Dyspnea'] )
# print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_0') )
# print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_1') )
# print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_0') )
# print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_1') )
# #print( asia_cpts['Xray'])
# print( asia_cpts['Xray']('TuberculosisOrCancer_0') )
# print( asia_cpts['Xray']('TuberculosisOrCancer_1') )

#asia_stretch = stretch(asia_model, graph_output=True,observed=False)

#graph_image(asia_stretch['graph'], "experiment2")

#print("\nAsia pointer:", asia_stretch['pointer'] )
#asia_joint = evaluate_stretch(asia_stretch['channels'])
#print("\nAsia final state:", asia_joint )

#print("\nAsia marginals:", marginals_stretch(asia_stretch))

#print( asia_inference.query(['Tuberculosis'])['Tuberculosis'] )
#print( asia_inference.query(['LungCancer'])['LungCancer'] )
#print( asia_inference.query(['Bronchitis'])['Bronchitis'] )
# print( asia_inference.query(['TuberculosisOrCancer'])['TuberculosisOrCancer'] )
# print( asia_inference.query(['Xray'])['Xray'] )
# print( asia_inference.query(['Dyspnea'])['Dyspnea'] )



print("\nAsia inference 1, by hand\n")

"""

asia_stretch = stretch(asia_model, graph_output=True,observed=True)
#asia_stretch = stretch(asia_model, graph_output=True,observed=False)

#graph_image(asia_stretch['graph'], "asia")

asia_joint = evaluate_stretch(asia_stretch['channels'])

asia_joint = reorder_state_domains(asia_joint,
                                   ['VisitToAsia', 
                                    'Tuberculosis', 
                                    'TuberculosisOrCancer', 
                                    'Xray', 
                                    'Dyspnea',
                                    'LungCancer',
                                    'Bronchitis',
                                    'Smoker'])

asia_domain = asia_joint.dom

p = point_pred('Tuberculosis_0', asia_domain[1])

p1 = truth(asia_domain[0]) @ p @ truth(asia_domain[2:])

print("* Via joint state:")
print( asia_joint / p1 % [0,0,0,0,1,0,0,0] )

print("* Via transformations:")
print( asia_cpts['Dyspnea'] \
       >> ((asia_cpts['Bronchitis'] @ asia_cpts['TuberculosisOrCancer']) \
           >> ((idn(asia_domain[7]) @ asia_cpts['LungCancer'] @ idn(asia_domain[1])) \
               >> ((copy(asia_domain[7]) @ idn(asia_domain[1])) \
                   >> (asia_cpts['Smoker'] @ ((asia_cpts['Tuberculosis'] >> asia_cpts['VisitToAsia']) / p))))) )


print("* Via variable elimination")
print( asia_inference.query(['Dyspnea'], 
                            evidence={'Tuberculosis': 0})['Dyspnea'] )

"""


print("\nAsia inference 2, automated\n")

asia_stretch = stretch(asia_model, observed=False)

print("\n* Via transformation-inference:")
print( inference_query(asia_stretch, 'Bronchitis', {'Xray' : [1,0], 'Tuberculosis' : [0,1]}) )

print("\n* Via variable elimination")
print( asia_inference.query(['Bronchitis'], evidence={'Xray': 0, 'Tuberculosis' : 1})['Bronchitis'] )

"""

N = 1

print("\nInference timing comparison,", N, "times\n")

t1 = timeit.timeit(lambda: 
                   asia_inference.query(['Bronchitis'], 
                                        evidence={'Xray': 0, 'Smoker' : 1})
                   ['Bronchitis'],
                   number = N)

t2 = timeit.timeit(lambda: 
                   inference_query(asia_stretch, 'Bronchitis', 
                                   {'Xray' : [1,0], 'Smoker' : [0,1]}),
                   number = N) 

print("Times for: variable elimination, transformations, fraction")
print(t1)
print(t2)
print(t1/t2)


"""
