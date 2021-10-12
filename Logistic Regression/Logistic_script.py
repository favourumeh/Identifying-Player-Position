# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:25:38 2021

@author: favou
"""
#Importing libraries
import pandas as pd 

from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
from sklearn.model_selection import cross_val_score, cross_validate


import matplotlib.pyplot as plt
import seaborn as sns

import math
import numpy as np
import scipy.stats as stats


# Importing raw data

    # train_test split
X_train = pd.read_csv('X_train_correctly_scaled.csv')
X_test = pd.read_csv('X_test_correctly_scaled.csv')

y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

    # Evaluation 
df_10 = pd.read_csv('df_10.csv')
df_10 = df_10.iloc[:,1:]
    

# Multicolinearity check
    
    #correlation plot (remove all where coeff>= 0.7)
hm = X_train.corr()
g = sns.heatmap(hm, cmap="YlGnBu", annot= False, fmt = '.1g', annot_kws = {'size':8}).set(title = 'Correlation plot of all independent variables')

X_train1 = X_train.drop(['FG', 'FGA','3P','2P', 'FT','ORB','DRB','eFG_per', 'PTS'], axis =1)

hm = X_train1.corr()
g = sns.heatmap(hm, cmap="YlGnBu", annot= True, fmt = '.1g', annot_kws = {'size':8}).set(title = 'Correlation plot of all independent variables')
    
    #VIF (remove all where VIF>= 5)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_t = deepcopy(X_train1)
X_t = add_constant(X_t) #add a constant term even though it isn't a predictor because statsmodel requires it 

pd.Series([variance_inflation_factor(X_t.values, i) for i in range(X_t.shape[1])], index=X_t.columns)

X_train = X_train[X_train1.columns]
X_test= X_test[X_train1.columns]


# Feature scalling function (minmaxscaler) 

def min_max_scaler(df):
    df5 = pd.DataFrame()
    for column in df:
        if 'per' not in column :
            df5[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
        
        if 'per' in column:
            df5[column] = df[column]
    return df5

# Modeling KNN 

    # Modelling 
log_reg = LogisticRegression(penalty = 'none', random_state=0, solver = 'lbfgs', max_iter=10000)
log_reg.fit(X_train, y_train.values.flatten())


        # Evaluate test dataset on metrics (Accuracy)
from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

ac_train = accuracy_score(y_pred, y_train) # 0.6921 accuracy
ac = accuracy_score(y_pred_test, y_test) # 0.6559 accuracy



        # Confusion Matrix 
def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, annot=True, fmt='.1f', annot_kws={'size':12})
    else:
        sns.heatmap(cm)
    plt.title(title)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')


cm = confusion_matrix(y_train, y_pred)
cm1 = confusion_matrix(y_test, y_pred_test)


plt.figure()
plot_confusion_matrix(cm,classes = log_reg.classes_, title='log_reg: Confusion matrix for trained data')
plt.figure()
plot_confusion_matrix(cm1, classes = log_reg.classes_, title='log_reg: Confusion matrix for test data')

         
        # classification report 
print(classification_report(y_train, y_pred)) #training data 

print(classification_report(y_test, y_pred_test)) # testing data

########################## ignore this section ################################

        # Cross validation

a = pd.read_csv('X_train_unscaled.csv')[X_train1.columns]
b = pd.read_csv('X_test_unscaled.csv')[X_train1.columns]

df_90 = pd.concat([a,b], axis =0)

X_s = min_max_scaler(df_90)

y = pd.concat([y_train, y_test], axis =0)


log_reg1 = LogisticRegression(penalty = 'none', random_state=0, solver = 'lbfgs', max_iter=5000)
Accuracy_scores = cross_val_score(log_reg1, X_s, y.values.flatten(), cv = 5, scoring = 'accuracy') # [0.69377163, 0.70025952, 0.70272609, 0.69190826, 0.70488966]
mean_Accuracy_score = Accuracy_scores.mean() # 0.683 slighly higher than accuracy of test set 0.6559 
print(Accuracy_scores)
###############################################################################
    


# Tunning Model 
    # 1A) Using gridsearch cross validation on training set
from sklearn.model_selection import GridSearchCV

log_reg2 = LogisticRegression()

param_grid = {'max_iter' : [5000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga']}
   
clf_log_reg = GridSearchCV(log_reg2, param_grid = param_grid, cv = 5, verbose=True, n_jobs = -1)
best_clf_log_reg = clf_log_reg.fit(X_train, y_train)
best_clf_log_reg.best_score_ # mean accuracy 0.691
best_clf_log_reg.best_params_ #{'C': 206.913808111479, 'max_iter': 5000, 'penalty': 'l2', 'solver': 'sag'}

        # Evaluating parameters derved from Gridsearch on test set

log_reg3 = LogisticRegression(max_iter = 5000, penalty = 'l2', C =206.91381, solver = 'sag')

log_reg3.fit(X_train, y_train.values.flatten())

y_pred = log_reg3.predict(X_train)
y_pred_test = log_reg3.predict(X_test)

ac_train = accuracy_score(y_pred, y_train) # 0.6926 accuracy
ac = accuracy_score(y_pred_test, y_test) # 0.6555 accuracy (slighly worse than before 0.6559 )


    #2A) Adjusting C to the one that maximises the accuracy of the test data

#(Result: best is C = 29.764 for mean accuracy (0.6571) and log_loss (0.8679))

C_list = np.logspace(-4, 4, 20)

acc_list = []
log_loss_list = []
acc_CV_list=[]
for C in C_list:
    
    log_reg4 = LogisticRegression(max_iter = 5000, penalty = 'l2', C =C, solver = 'sag')
    log_reg4.fit(X_train, y_train.values.flatten())
    
    acc_score = log_reg4.score(X_test, y_test)
    acc_list.append(acc_score)
     
    ll = log_loss(y_test.values.flatten(), log_reg4.predict_proba(X_test))
    log_loss_list.append(ll)

metrics_df = pd.DataFrame(zip(C_list,acc_list, log_loss_list), columns = ['C', 'Accuracy', 'log_loss'])



    #3A) Adjusting C to the one that maximises the CV accuracy of the train data

#(Result: best is C = 206.914 for mean accuracy (0.6915) and log_loss (0.7512) )
#This result should be more generaliseable than the one calculated above 

a = []
b = []

for C in C_list:
    log_reg5 = LogisticRegression(max_iter = 5000, penalty = 'l2', C =C, solver = 'sag')
    metrics1 = cross_validate(log_reg5, X_train, y_train.values.flatten(), cv = 5, scoring = ['accuracy', 'neg_log_loss'])
    a.append(np.mean(metrics1['test_accuracy']))
    b.append(-np.array(metrics1['test_neg_log_loss']).mean())

metrics_df2 = pd.DataFrame(zip(C_list, a, b), columns = ['C', 'mean_accuracy', 'mean_log_loss'])


    #4A) Adjusting C to the one that maximises the CV accuracy of the 90% data
#Result: C =29.7635 ,accuracy = 0.689626
a1 = []
b1 = []

for C in C_list:
    log_reg6 = LogisticRegression(max_iter = 5000, penalty = 'l2', C =C, solver = 'sag')
    metrics1 = cross_validate(log_reg6, X_s, y.values.flatten(), cv = 5, scoring = ['accuracy', 'neg_log_loss'])
    a1.append(np.mean(metrics1['test_accuracy']))
    b1.append(-np.array(metrics1['test_neg_log_loss']).mean())

metrics_df3 = pd.DataFrame(zip(C_list, a1, b1), columns = ['C', 'mean_accuracy', 'mean_log_loss'])



#Evaluating model 

a = pd.read_csv('X_train_unscaled.csv')[X_train1.columns]
b = pd.read_csv('X_test_unscaled.csv')[X_train1.columns]

df_90 = pd.concat([a,b], axis =0)

X_s = min_max_scaler(df_90)

y = pd.concat([y_train, y_test], axis =0)
    
# # Evaluating model for df_10 (using 3a results)
# X_eval = df_10.drop('Pos', axis = 1)[X_train1.columns]

# X_eval = min_max_scaler(X_eval)
# y_eval = df_10.Pos

# log_reg_eval = LogisticRegression(max_iter = 5000, penalty = 'l2', C = 206.914, solver = 'sag')
# log_reg_eval.fit(X_train, y_train.values.flatten()) 

# y_pred_eval = log_reg_eval.predict(X_eval)
# y_pred_test = log_reg_eval.predict(X_test)


#       #Accuracy
# ac_eval = accuracy_score(y_pred_eval, y_eval) # 0.653 accuracy
# ac_test = accuracy_score(y_pred_test, y_test) # 0.656 accuracy


# Evaluating model for df_10 (using 4a results)
X_eval = df_10.drop('Pos', axis = 1)[X_train1.columns]

X_eval = min_max_scaler(X_eval)
y_eval = df_10.Pos

log_reg_eval1 = LogisticRegression(max_iter = 5000, penalty = 'l2', C =29.7635, solver = 'sag')
log_reg_eval1.fit(X_s, y.values.flatten()) 


y_pred_eval = log_reg_eval1.predict(X_eval)


        #Accuracy
ac_eval = accuracy_score(y_pred_eval, y_eval) # 0.649 accuracy

            #classifcation report
print(classification_report(y_eval, y_pred_eval))
            #confusion matrix 
cm2 = confusion_matrix(y_eval, y_pred_eval)


plt.figure()
plot_confusion_matrix(cm2, classes = log_reg_eval1.classes_, title='log_reg: Confusion matrix for evaluation data')



###################### Saving Model ###########################################

# #saving model 
# import pickle

# with open('Optimal_log_reg_model.sav','wb') as f:
#       pickle.dump(log_reg_eval1,f)
      
###############################################################################