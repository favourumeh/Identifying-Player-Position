# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:09:37 2021

Ensemble Approach 

@author: favou
"""

import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Load Models 

    #KNN
with open('Optimal_KNN_model_train.sav', 'rb') as pickle_file:
     knn = pickle.load(pickle_file)

    #log_reg
with open('Optimal_log_reg_model_train.sav', 'rb') as pickle_file:
     log_reg = pickle.load(pickle_file)
     
    #GNB(takes unscaled data)
with open('Optimal_GNB_model_train.sav', 'rb') as pickle_file:
     GNB = pickle.load(pickle_file)
    
    
# Import 'training' set

    # scaled features (for KNN and log_reg)
X_test_s = pd.read_csv('X_test_correctly_scaled.csv')
    
    # unscaled features (for GNB)
X_test = pd.read_csv('X_test_unscaled.csv')

y_test = pd.read_csv('y_test.csv')

# Features taken by each model 

    #for KNN
headings_knn = ['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']

    #for log_reg
headings_log_reg = ['FG_per', '3PA', '3P_per', '2PA', '2P_per', 'FTA', 'FT_per', 'TRB',
            'AST', 'STL', 'BLK', 'TOV', 'PF']  

    #for naive bayes 
headings_GNB = ['FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per', 'FTA',
            'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']



# Ensemble function

#def Ensemble(X_df_list,   ):

    # Prediction 
y_knn = knn.predict(X_test_s[headings_knn])
y_log_reg = log_reg.predict(X_test_s[headings_log_reg])
y_GNB = GNB.predict(X_test[headings_GNB])


    # Prediction probabilities 
KNN_proba = pd.read_csv('Pred_prob_test_KNN.csv')
Log_reg_proba = pd.read_csv('Pred_prob_test_log_reg.csv')
GNB_proba = pd.read_csv('Pred_prob_test_GNB.csv')



    #Creating Column of Actual label, Model Predecition and Voting splits (3-0, 2-1, 0-3)
Pred_df = pd.DataFrame(zip(y_test.values.flatten(), y_knn, y_log_reg, y_GNB), columns = ['Actual','knn_pred', 'log_reg_pred', 'GNB_pred'])
Pred_df['knn_pred=log_reg_pred'], Pred_df['knn_pred=GNB_pred'], Pred_df['log_reg_pred=GNB_pred'] =0,0,0 
Pred_df['knn_pred=log_reg_pred'][Pred_df['knn_pred'] == Pred_df['log_reg_pred']] =1 
Pred_df['knn_pred=GNB_pred'][Pred_df['knn_pred'] == Pred_df['GNB_pred']] =1 
Pred_df['log_reg_pred=GNB_pred'][Pred_df['log_reg_pred'] == Pred_df['GNB_pred']] =1 
Pred_df['sum'] = Pred_df['knn_pred=log_reg_pred']+ Pred_df['knn_pred=GNB_pred']+ Pred_df['log_reg_pred=GNB_pred']




######################### Selecting the model of choice #######################

###drop existing columns if necessary


    #1) Concensus and 2-to-1 split cases
Pred_df['Model_of_choice'] = 0


        # Generating indices  

            #case: 'knn_pred=log_reg_pred'
indices_knn_log = Pred_df[Pred_df['knn_pred=log_reg_pred']== 1].index

            #case: 'knn_pred=GNB_pred'
indices_knn_GNB = Pred_df[Pred_df['knn_pred=GNB_pred']== 1].index

            #case: 'log_reg_pred =GNB_pred'
indices_log_GNB = Pred_df[Pred_df['log_reg_pred=GNB_pred']== 1].index


        # Asigning a module of choice for the Concensus and 2-to-1 split cases
Pred_df['Model_of_choice'][indices_log_GNB] = 'GNB_pred'
Pred_df['Model_of_choice'][indices_knn_GNB]= 'knn_pred'
Pred_df['Model_of_choice'][indices_knn_log] = 'knn_pred'


    #2) 'hung' parliament (all disagree) case

        # indices of 'hung'
indices_hung = Pred_df[Pred_df['sum'] == 0].index

###############################################################################





##################### Ensemble prediction 1: Hard Voting 1.0 ##################
        # import random module
import random

        #2A) Randomly asigning a module of choice for the 'hung' parliament (all disagree) case

options = ['knn_pred', 'log_reg_pred', 'GNB_pred']

rand_opt = np.array([random.choice(options) for i in range(len(indices_hung))]).reshape(-1,1) # randomly choses from 'options' list

Pred_df['Model_of_choice'][indices_hung] = rand_opt.flatten()

        # prediction
Pred_df['Ensemble_pred'] = 0

for i,mod_choice in enumerate(Pred_df['Model_of_choice']):
    Pred_df.loc[i, 'Ensemble_pred'] = Pred_df.loc[i, mod_choice] 


y_pred_ensemble = np.array(Pred_df['Ensemble_pred'])

###############################################################################

#$$$$$$$$$ Brief Analysis of All cases (Concensus, hung, and Majority ) $$$$$$

All_wrong = Pred_df[(Pred_df['sum'] == 3) & (Pred_df.Incorrect == 1)]
Two_one_split = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.Incorrect == 1)]
hung_split = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Incorrect == 1)]

len(All_wrong) +len(Two_one_split) + len(hung_split)

    #Part 1: looking at concensus cases (54.8% of all cases)
100*len(Pred_df[Pred_df['sum'] == 3])/len(Pred_df)

        #35.18% of all incorrect predictions have a concesus case 
            #22.54% of all concensus cases are wrong; 
                #Wrong concensus cases make up 12.4% of all cases
100*len(All_wrong)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(All_wrong)/len(Pred_df[(Pred_df['sum'] == 3)])     
100*len(All_wrong)/len(Pred_df)

    # Part 2: looking at hung cases (3.317% of all cases): 
100*len(Pred_df[Pred_df['sum'] == 0])/len(Pred_df)

        #7.07% of all incorrect cases is belong to 'hung' scenario;
            #64.28% of all 'hung' are wrongly predicted; 
                #Wrong hung cases make up 2.45% of all cases
100*len(hung_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(hung_split)/len(Pred_df[(Pred_df['sum'] == 0)]) 
100*len(hung_split)/len(Pred_df) 

    # Part 3: looking at majority vote(2-1) cases (41.3% of all cases):
100*len(Pred_df[Pred_df['sum'] == 1])/len(Pred_df)
        
        #57.8% of all incorrect cases is belong to 2-1 scenario:
            #49.1% of all 2-1 splits are wrongly predicted; 
                #Wrong 2-1 cases make up 20.3% of all cases )
100*len(Two_one_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(Two_one_split)/len(Pred_df[(Pred_df['sum'] == 1)])

100*len(Two_one_split)/len(Pred_df)


#Comments and Questions 

    #1) Consensus:
        #If all models come to a concensus on a player's position then it maybe safe
        #to presume that that player 'plays' like the position predicted. In this
        #situation the player's actual position assignment is dubious. Humans are not
        #always logical in their categorisation
    
    #2) 2-1 split:
        #Any imporvements to the model should be focused this scenario as 57.8% of
        #all incorrect predictions involve this case (thus the potential increase
        #in accuracy is as much as 20.3%)
        
        #However, if two models have a concensus on a player's position it could be
        #that the player plays that way
    
    #3) The hung cases should not just be decided randomly. Though small in number
        # the randomness of the selection process mean that the model's answer 
        # changes with each run even if the overall accuracy hardly changes.  

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#################### Addressing the 'hung' cases ###############################
Hung_cases  = Pred_df[(Pred_df['sum'] == 0)]

    # Hung cases where no model predicted correctly:
Hung_All_wrong = Hung_cases[(Hung_cases.Actual != Hung_cases.knn_pred) & (Hung_cases.Actual != Hung_cases.log_reg_pred) & (Hung_cases.Actual != Hung_cases.GNB_pred)] 

        # there are 7 'lost cause' hung cases (i.e. where no model got the correct answer)
            # this is 0.27% of all test data
len(Hung_All_wrong)
100*len(Hung_All_wrong)/len(Pred_df)

    # Finding the model with the best precision for hung cases
columns = ['Position', 'no. of knn_preds(% precision)', 'no. of log_reg_preds(% precision)', 'no of GNB_preds(% precision)']
columns1 = ['Position', 'No. of correct knn_pred', 'No. of correct  log_reg_pred', 'No. of correct  GNB_pred']

Model_Pres_df = pd.DataFrame(columns =columns )
Model_Pres_df1 = pd.DataFrame(columns =columns1 )


#if a model predicts a label what are the chances that it is correct 

for i in ['SG', 'PG', 'SF', 'PF', 'C']:
    
    mods = ['knn_pred', 'log_reg_pred', 'GNB_pred']
    
    # plt.figure()
    pres_list = []
    pres_list1 = []
    pres_list2 = []
    
    for mod in mods: 
        A1 = Hung_cases[(Hung_cases[mod] == i)]
        A1a = Hung_cases[(Hung_cases[mod] == i) & (Hung_cases.Actual == i)]
        no_pred = len(A1)
        pres = round(100*len(A1a)/len(A1), 2)
        pres1 = len(A1a)/len(A1)
        pres_list.append(f'{no_pred}({pres})')
        pres_list1.append(pres)
        pres_list2.append(pres1*no_pred)

    # Pres_plot_df = pd.DataFrame(zip(mods, pres_list1), columns = ['Models','Accuracy'])
    # ax = sns.barplot(x="Models", y= 'Accuracy', data=Pres_plot_df)
    # plt.ylabel(f'Proportion of {i} predictions')
    # plt.title(f'When the actual position = {i}')
    Model_Pres_df = Model_Pres_df.append(pd.Series(np.array([i]+pres_list), index = columns), ignore_index=True) 
    Model_Pres_df1 = Model_Pres_df1.append(pd.Series(np.array([i]+pres_list2), index = columns1), ignore_index=True) 
        

Model_Pres_df1['No. of correct knn_pred'] = pd.to_numeric(Model_Pres_df1['No. of correct knn_pred'] , errors = 'coerce')
Model_Pres_df1['No. of correct  log_reg_pred'] = pd.to_numeric(Model_Pres_df1['No. of correct  log_reg_pred'] , errors = 'coerce')
Model_Pres_df1['No. of correct  GNB_pred'] = pd.to_numeric(Model_Pres_df1['No. of correct  GNB_pred'] , errors = 'coerce')


    #[Findings: there are 98 'hung' cases 
                # 91 'non-lost cause' cases and 7 lost-cause(i.e. no model got the right answer)  
                # No. of correct knn_pred                18
                # No. of correct  log_reg_pred           44
                # No. of correct  GNB_pred               29]
                
    #[Conclusion: In hung cases go for a log_reg prediction because out of all 
    #            'non-lost cause' hung cases it had a 48% chance of getting 
    #             the correct answer and for all lost cause cases it had a 45% 
    #             chance of getting the correct answer ]


##############################################################################




##################### Ensemble prediction 2: Playing the odds #################
    
    #Determing the accuracy performace for each model for each label in a hung parliament scenrio
    
# A = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == 'SG' )]
# B = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == 'PG' )]
# C = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == 'SF' )]
# D = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == 'PF' )]
# E = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == 'C' )]

columns = ['Position', 'No_cases', 'knn_pred', 'log_reg_pred', 'GNB_pred']
Model_Acc_df = pd.DataFrame(columns =columns )

for i in ['SG', 'PG', 'SF', 'PF', 'C']:
    
    plt.figure()
    acc_list = []
    no_hung = len(Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == i ) ])
    mods = ['knn_pred', 'log_reg_pred', 'GNB_pred']
    for j in mods:
        mod_correct = len(Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Actual == i ) & (Pred_df[j]== i ) ])
        acc_list.append(100*mod_correct/no_hung)
        
    Acc_plot_df = pd.DataFrame(zip(mods, acc_list), columns = ['Models','Accuracy'])
    ax = sns.barplot(x="Models", y= 'Accuracy', data=Acc_plot_df)
    plt.ylabel(f'Proportion of {i} predictions')
    plt.title(f'When the actual position = {i}')
    
    Model_Acc_df = Model_Acc_df.append(pd.Series(np.array([i]+[no_hung]+acc_list), index = columns), ignore_index=True) 
    
    
Model_Acc_df.knn_pred = pd.to_numeric(Model_Acc_df.knn_pred , errors = 'coerce')
Model_Acc_df.log_reg_pred = pd.to_numeric(Model_Acc_df.log_reg_pred , errors = 'coerce')
Model_Acc_df.GNB_pred = pd.to_numeric(Model_Acc_df.GNB_pred , errors = 'coerce')

        #conclusion:
            #if GNB choses SG (i.e. GNB_SG) go with that 
            #if Log_reg_SF go with that if no GNB_SG
            #if KNN_PF go with that unless no GNB_SG or Log_reg_SF
            #if Log_reg_PG go with that unless no GNB_SG or Log_reg_SF or KNN_PF
            #if KNN_C go with that unless GNB_SG,  Log_reg_SF, Log_reg_PG
            
    #Predictions
for i in indices_hung:
    KNN_pred = Pred_df.loc[i, 'knn_pred' ]
    Log_reg_pred = Pred_df.loc[i, 'log_reg_pred' ]
    GNB_pred = Pred_df.loc[i, 'GNB_pred' ]

    # if GNB_pred == 'SG': # old hierarchy 
    #     Pred_df['Model_of_choice'][i] = 'GNB_pred'
            
    # if Log_reg_pred == 'SF':
    #     if GNB_pred != 'SG' :
    #         Pred_df['Model_of_choice'][i] = 'log_reg_pred'
    
    if Log_reg_pred == 'SF':
         Pred_df['Model_of_choice'][i] = 'log_reg_pred'
            
            
    if GNB_pred == 'SG':
        if Log_reg_pred != 'SF' :        
            Pred_df['Model_of_choice'][i] = 'GNB_pred'
    
        
    if KNN_pred == 'PF':
        if GNB_pred != 'SG' or Log_reg_pred != 'SF' :
            Pred_df['Model_of_choice'][i] = 'knn_pred'
    
    if Log_reg_pred == 'PG':
        if GNB_pred != 'SG' or KNN_pred != 'PF' :
            Pred_df['Model_of_choice'][i] = 'log_reg_pred'
 
    if KNN_pred == 'C':
        if GNB_pred != 'SG' or Log_reg_pred != 'SF' or Log_reg_pred != 'PG' :
            Pred_df['Model_of_choice'][i] = 'knn_pred'
    

    #remaining 'hung' cases  
rem_hung_indices = Pred_df[Pred_df['Model_of_choice']==0].index
Rem_hung_df = Pred_df.loc[rem_hung_indices, ['knn_pred', 'log_reg_pred', 'GNB_pred']]

pred_temp = pd.DataFrame(zip(Rem_hung_df['knn_pred'].values, Rem_hung_df['log_reg_pred'].values, Rem_hung_df['GNB_pred']), columns = 3*['A'] )

mode_pred_elements = pd.concat([pred_temp.iloc[:,0],pred_temp.iloc[:,1], pred_temp.iloc[:,2]], axis =0 ).mode()

    #Chose the model that best predicts the modal prediction (most frequent)
if len(mode_pred_elements) >1:    
    #if bimodal or more  then randomly choose a mode from modes 
    
    mode_pred_elements = random.choice(list(mode_pred_elements))

elif len(mode_pred_elements) == 1:
    if mode_pred_elements[0] == 'PG': #knn has best precision for PG
         Pred_df['Model_of_choice'][rem_hung_indices] = 'knn_pred'
         
    if mode_pred_elements[0] == 'C': #knn has best precision for C
        Pred_df['Model_of_choice'][rem_hung_indices] = 'knn_pred'
        
    if mode_pred_elements[0] == 'PF': #... has best precision for PF
        Pred_df['Model_of_choice'][rem_hung_indices] = 'GNB_pred'
    
    if mode_pred_elements[0] == 'SF': #... has best precision for SF
        Pred_df['Model_of_choice'][rem_hung_indices] = 'log_reg_pred'    
    
    if mode_pred_elements[0] == 'SG': #... has best precision for SG
        Pred_df['Model_of_choice'][rem_hung_indices] = 'log_reg_pred'    

    

for i,mod_choice in enumerate(Pred_df['Model_of_choice']):
    Pred_df.loc[i, 'Ensemble_pred'] = Pred_df.loc[i, mod_choice] 


y_pred_ensemble = np.array(Pred_df['Ensemble_pred'])

Pred_df['Incorrect'] = 0
Pred_df['Incorrect'][Pred_df.Actual != Pred_df.Ensemble_pred] =1
###############################################################################


#$$$$$$$$$ Brief Analysis of All cases (Concensus, hung, and Majority ) $$$$$$

All_wrong = Pred_df[(Pred_df['sum'] == 3) & (Pred_df.Incorrect == 1)]
Two_one_split = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.Incorrect == 1)]
hung_split = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Incorrect == 1)]

len(All_wrong) +len(Two_one_split) + len(hung_split)

    #Part 1: looking at concensus cases (54.8% of all cases)
100*len(Pred_df[Pred_df['sum'] == 3])/len(Pred_df)

        #35.18% of all incorrect predictions have a concesus case 
            #22.54% of all concensus cases are wrong; 
                #Wrong concensus cases make up 12.4% of all cases
100*len(All_wrong)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(All_wrong)/len(Pred_df[(Pred_df['sum'] == 3)])     
100*len(All_wrong)/len(Pred_df)

    # Part 2: looking at hung cases (3.317% of all cases): 
100*len(Pred_df[Pred_df['sum'] == 0])/len(Pred_df)

        #7.07% of all incorrect cases is belong to 'hung' scenario;
            #64.28% of all 'hung' are wrongly predicted; 
                #Wrong hung cases make up 2.45% of all cases
100*len(hung_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(hung_split)/len(Pred_df[(Pred_df['sum'] == 0)]) 
100*len(hung_split)/len(Pred_df) 

    # Part 3: looking at majority vote(2-1) cases (41.3% of all cases):
100*len(Pred_df[Pred_df['sum'] == 1])/len(Pred_df)
        
        #57.8% of all incorrect cases is belong to 2-1 scenario:
            #49.1% of all 2-1 splits are wrongly predicted; 
                #Wrong 2-1 cases make up 20.3% of all cases )
100*len(Two_one_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(Two_one_split)/len(Pred_df[(Pred_df['sum'] == 1)])

100*len(Two_one_split)/len(Pred_df)


#Comments and Questions 

    #1) Consensus:
        #If all models come to a concensus on a player's position then it maybe safe
        #to presume that that player 'plays' like the position predicted. In this
        #situation the player's actual position assignment is dubious. Humans are not
        #always logical in their categorisation
    
    #2) 2-1 split:
        #Any imporvements to the model should be focused this scenario as 57.8% of
        #all incorrect predictions involve this case (thus the potential increase
        #in accuracy is as much as 20.3%)
        
        #However, if two models have a concensus on a player's position it could be
        #that the player plays that way

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


####################### Evaluating Ensemble model #############################

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



    #1) Accuracy
ac_test = accuracy_score(y_test, y_pred_ensemble) # 0.649 accuracy  # 0.652 (improved hierarchy)          

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
plot_confusion_matrix(cm,classes = log_reg.classes_, title='Ensemble: Confusion matrix for test data')

    #3) Classification report 
print(classification_report(y_test, y_pred_ensemble))            

################################################################################




######################## Further Actions to model #############################
# SF accuracy is pultry becasue the we are overpredicting PF in place of SF which is in turn 
# reducing the precesion of PF and the overall accuracy of the model 

    #2-1 cases where PF was predicted and GNB predicted SF
C4a = Pred_df[(Pred_df['sum']==1) & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.GNB_pred =='SF') ]

    #2-1 cases where he majority descision of PF was actually correct
C4b = Pred_df[(Pred_df['sum']==1) & (Pred_df.GNB_pred =='SF') & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.Incorrect == 0)]
        # 34% of the timeThe majority decision was correct 
        
100*len(C4b)/len(C4a)

    #2-1 cases where PF was predicted whilst GNB correctly predicted SF
C4c = Pred_df[(Pred_df['sum']==1)  & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.GNB_pred == 'SF') &  (Pred_df.GNB_pred == Pred_df.Actual)]

        # 49.8% of the time GNB was correct
100*len(C4c)/len(C4a)

#[CONCLUSION: If the ENSEMBLE MODEL PREDICTS PF in a Majority vote and GNB predicts SF 
#             go with GNB's prediction because the Majority vote is only correct
#             34.2% (88 more correct results) of the time in this case whilst GNB 
#             is correct 49.8% (128 more correct results) of the time. The net 
#             effect is 40 more correct results which would increase the model's
#             accuracy by 1.56%]
    
 
C4b = Pred_df[(Pred_df['sum']==1) & (Pred_df.log_reg_pred =='SF') & (Pred_df.Ensemble_pred == 'PF') ]

100*len(C4a)/len(C4b)

    #2-1 case PF predictions 



##############################################ignore ########################
   
    #...their X values 
X_s_rem_hung = X_test_s.loc[rem_hung_indices, :]
X_rem_hung = X_test.loc[rem_hung_indices, :]    
    #...their probabilities  ********************* TBC ***********************
# Hung_KNN_prob = knn.predict_proba(X_s_rem_hung[headings_knn]) 
# Hung_Log_reg_proba = log_reg.predict_proba(X_s_rem_hung[headings_log_reg]) 
# Hung_GNB_proba = GNB.predict_proba(X_rem_hung[headings_GNB]) 

# Hung_Log_reg_proba = pd.DataFrame(log_reg.predict_proba(X_s_rem_hung[headings_log_reg]), columns = ['Log_reg_'+ i for i in list(log_reg.classes_)])

Hung_KNN_prob = pd.read_csv('Pred_prob_test_KNN.csv').loc[rem_hung_indices, :]
Hung_Log_reg_proba = pd.read_csv('Pred_prob_test_log_reg.csv').loc[rem_hung_indices, :]
Hung_GNB_proba = pd.read_csv('Pred_prob_test_GNB.csv').loc[rem_hung_indices, :]


############################################################



######################## IGNORE: Further Actions to model #############################
# SF accuracy is pultry becasue the we are overpredicting PF which is in turn 
# reducing the accuracy of the model 

#     # All Cases: Where we incorrectly predicted something else when the reality was SF
    
# C1 = Pred_df[(Pred_df.Incorrect ==1) & (Pred_df.Actual == 'SF') ]

#     #2-1 cases: Where we incorrectly predicted something else when the reality was SF
# Two_one_wrong_SF = Pred_df[(Pred_df.Incorrect ==1) & (Pred_df.Actual == 'SF') & (Pred_df['sum']==1)]
# Two_one_wrong_SF_PF = Pred_df[(Pred_df.Incorrect ==1) & (Pred_df.Actual == 'SF') & (Pred_df['sum']==1) & (Pred_df.Ensemble_pred == 'PF')]

#         #68.04% of all wrong SF predictions were made when we had a 2-1 case
# 100*len(Two_one_wrong_SF)/ len(C1)

#         # 72 % of the time PF was predicted when SF should have been predicted for 2-1 case 
# 100*len(Two_one_wrong_SF_PF)/len(Two_one_wrong_SF)


#     #2-1 cases when GNB was correctly predicted SF whilst the Majority vote was PF
# C2 = Two_one_wrong_SF[(Two_one_wrong_SF.Actual == Two_one_wrong_SF.GNB_pred) & (Two_one_wrong_SF.Ensemble_pred == 'PF') ]
          
#       # ~60% of the time GNB's SF prediction was correct when majority vote was wrong ***********
#                 #100% of the time the incorrect Majority vote was PF when really this should be PF (just look at C2)
# 100*len(C2)/len(Two_one_wrong_SF)

#     # 2-1 cases where PF was predicted whilst GNB predicted SF 
# C3 = Pred_df[(Pred_df['sum']==1) & (Pred_df.GNB_pred =='SF') & (Pred_df.Ensemble_pred == 'PF') ]

#         # For this situation GNB would have been correct 49.8% of the time
# 100*len(C2)/len(C3)

#     #2-1 cases where the Majority vote of PF was correct whilst GNB chose SF
# C3B = Pred_df[(Pred_df['sum']==1) & (Pred_df.GNB_pred =='SF') & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.Incorrect == 0)]

#         #34.2% of the time the Majority vote of PF was correct whilst GNB chose SF *****************
# 100*len(C3B)/len(C3)


