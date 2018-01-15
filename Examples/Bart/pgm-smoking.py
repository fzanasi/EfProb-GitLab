from pgm_efprob import *

print("\nSAC = Smoking-Ashtray-Cancer example")
print("====================================\n")

sac_model = BayesianModel([('Smoking', 'Ashtray'), 
                           ('Smoking', 'Cancer')])

cpd_smoking = TabularCPD(variable='Smoking', variable_card=2,
                         values=[[0.3], [0.7]])

cpd_ashtray = TabularCPD(variable='Ashtray', variable_card=2,
                      values=[[0.95, 0.25], [0.05, 0.75]],
                      evidence=['Smoking'], evidence_card=[2])

cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.4, 0.05], [0.6, 0.95]],
                        evidence=['Smoking'], evidence_card=[2])

sac_model.add_cpds(cpd_smoking, cpd_ashtray, cpd_cancer)

print("SAC model correct: ", sac_model.check_model() )
print ("\nNodes: ", sac_model.nodes())
print("\nTables:")
print (sac_model.get_cpds('Smoking'))
print (sac_model.get_cpds('Ashtray'))
print (sac_model.get_cpds('Cancer'))

sac_inference = VariableElimination(sac_model)

print("\nMarginals:")
print( sac_inference.query(['Smoking']) ['Smoking'])
print( sac_inference.query(['Cancer']) ['Cancer'])
print( sac_inference.query(['Ashtray']) ['Ashtray'])

print("\nConditionals (note: 0 stands for true-case)")
print( sac_inference.query(['Cancer'], evidence={'Smoking': 0}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Smoking': 1}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Ashtray': 0}) ['Cancer'])
print( sac_inference.query(['Cancer'], evidence={'Ashtray': 1}) ['Cancer'])


print("\nGraphs and channels:\n")

sac_graph = pydot_graph_of_pgm(sac_model)

#graph_image(sac_graph, "sac")

sac_cpts = efprob_channels_of_pgm(sac_model)

print( sac_cpts["Smoking"] )
#print( sac_cpts["Ashtray"] )
print( sac_cpts["Ashtray"]("Smoking_0") )
print( sac_cpts["Ashtray"]("Smoking_1") )
#print( sac_cpts["Cancer"] )
print( sac_cpts["Cancer"]("Smoking_0") )
print( sac_cpts["Cancer"]("Smoking_1") )



print("SAC stretch output, ordinary and observed\n")

sac_stretch = stretch(sac_model)

#graph_image(sac_stretch['graph'], "sac")

print("\nSAC state:", evaluate_stretch(sac_stretch['channels']) )

sac_joint_stretch = stretch(sac_model, observed=True)

#graph_image(sac_joint_stretch['graph'], "sac")

sac_joint = evaluate_stretch(sac_joint_stretch['channels'])

print("\nSAC joint state:", sac_joint )

print("\nMAP inference", (sac_joint % [1,1,1]).MAP() )

sac_inference = VariableElimination(sac_model)
print( sac_inference.map_query(variables=['Cancer', 'Ashtray', 'Smoking']) )
print( sac_inference.map_query(variables=['Cancer', 'Smoking']) )
print( sac_inference.map_query(variables=['Smoking', 'Ashtray']) )


