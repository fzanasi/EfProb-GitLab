from pgm_efprob import *

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
                    # [[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]
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

asia_inference = VariableElimination(asia_model)

asia_graph = pydot_graph_of_pgm(asia_model)
#graph_image(asia_graph, "asia")

asia_cpts = efprob_channels_of_pgm(asia_model)

print( asia_cpts['Smoker'] )
print( asia_cpts['VisitToAsia'] )
print( asia_cpts['LungCancer']('Smoker_0') )
print( asia_cpts['LungCancer']('Smoker_1') )
#print( asia_cpts['Tuberculosis'] )
print( asia_cpts['Tuberculosis']('VisitToAsia_0') )
print( asia_cpts['Tuberculosis']('VisitToAsia_1') )
#print( asia_cpts['Bronchitis'] )
print( asia_cpts['Bronchitis']('Smoker_0') )
print( asia_cpts['Bronchitis']('Smoker_1') )
#print( asia_cpts['TuberculosisOrCancer'] )
print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_0') )
print( asia_cpts['TuberculosisOrCancer']('LungCancer_0', 'Tuberculosis_1') )
print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_0') )
print( asia_cpts['TuberculosisOrCancer']('LungCancer_1', 'Tuberculosis_1') )
#print( asia_cpts['Dyspnea'] )
print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_0') )
print( asia_cpts['Dyspnea']('Bronchitis_0', 'TuberculosisOrCancer_1') )
print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_0') )
print( asia_cpts['Dyspnea']('Bronchitis_1', 'TuberculosisOrCancer_1') )
#print( asia_cpts['Xray'])
print( asia_cpts['Xray']('TuberculosisOrCancer_0') )
print( asia_cpts['Xray']('TuberculosisOrCancer_1') )

# 0.5|Smoker_0> + 0.5|Smoker_1>
# 0.01|VisitToAsia_0> + 0.99|VisitToAsia_1>
# 0.1|LungCancer_0> + 0.9|LungCancer_1>
# 0.01|LungCancer_0> + 0.99|LungCancer_1>
# 0.05|Tuberculosis_0> + 0.95|Tuberculosis_1>
# 0.01|Tuberculosis_0> + 0.99|Tuberculosis_1>
# 0.6|Bronchitis_0> + 0.4|Bronchitis_1>
# 0.3|Bronchitis_0> + 0.7|Bronchitis_1>
# 1|TuberculosisOrCancer_0> + 0|TuberculosisOrCancer_1>
# 0|TuberculosisOrCancer_0> + 1|TuberculosisOrCancer_1>
# 0|TuberculosisOrCancer_0> + 1|TuberculosisOrCancer_1>
# 0|TuberculosisOrCancer_0> + 1|TuberculosisOrCancer_1>
# 0.9|Dyspnea_0> + 0.1|Dyspnea_1>
# 0.7|Dyspnea_0> + 0.3|Dyspnea_1>
# 0.8|Dyspnea_0> + 0.2|Dyspnea_1>
# 0.1|Dyspnea_0> + 0.9|Dyspnea_1>
# 0.98|Xray_0> + 0.02|Xray_1>
# 0.05|Xray_0> + 0.95|Xray_1>

