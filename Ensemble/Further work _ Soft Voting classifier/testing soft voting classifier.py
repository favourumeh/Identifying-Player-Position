# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:03:14 2021

@author: favou
"""
import pandas as pd

KNN_prob = pd.read_csv('Pred_prob_test_KNN.csv')[['knn_C', 'knn_PF', 'knn_PG', 'knn_SF', 'knn_SG']]
log_reg_prob = pd.read_csv('Pred_prob_test_log_reg.csv')[['Log_reg_C', 'Log_reg_PF', 'Log_reg_PG', 'Log_reg_SF', 'Log_reg_SG']]
GNB_prob = pd.read_csv('Pred_prob_test_GNB.csv')[['GNB_C', 'GNB_PF', 'GNB_PG', 'GNB_SF', 'GNB_SG']]


SV_prob = pd.DataFrame()

positions = ['C', 'PF', 'PG', 'SF', 'SG']
for i in positions:
    SV_prob[i] = (KNN_prob['knn_'+i] + log_reg_prob['Log_reg_'+i] + GNB_prob['GNB_'+i])/3
    
    
SV_prob['Pred'] = 0
SV_prob.drop('Pred', axis = 1, inplace = True)

for j in range(len(SV_prob)):
    max_r = SV_prob.loc[j, :].max()
    SV_prob.loc[j, 'Pred'] =  SV_prob.loc[j, SV_prob.loc[j,:] == max_r].index[0]
    
y_pred_ensemble = SV_prob.Pred

y_test = pd.read_csv('Pred_prob_test_KNN.csv').Actual
SV_prob['Actual'] =  pd.read_csv('Pred_prob_test_KNN.csv').Actual

A = SV_prob[SV_prob.Pred != SV_prob.Actual]
B = A[A.Actual == 'PF']
B1 = A[A.Actual == 'SF']



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


ac_test = accuracy_score(y_test, y_pred_ensemble) # 0.647 accuracy  #0.668

    #2) Confusion Matrix
def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, annot=True, fmt='.1f', annot_kws={'size':12})
    else:
        sns.heatmap(cm)
    plt.title(title)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')


cm = confusion_matrix(y_test, y_pred_ensemble) 


plt.figure()
plot_confusion_matrix(cm,classes = ['C', 'PF', 'PG', 'SF', 'SG'], title='Ensemble_sv: Confusion matrix for test data')

    #3) Classification report 
print(classification_report(y_test, y_pred_ensemble))   