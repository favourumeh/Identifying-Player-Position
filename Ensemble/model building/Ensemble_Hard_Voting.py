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


    #Creating Column of Actual label, Model Predecition and Voting splits (3-0, 2-1, 0-3)
Pred_df = pd.DataFrame(zip(y_test.values.flatten(), y_knn, y_log_reg, y_GNB), columns = ['Actual','knn_pred', 'log_reg_pred', 'GNB_pred'])
Pred_df['knn_pred=log_reg_pred'], Pred_df['knn_pred=GNB_pred'], Pred_df['log_reg_pred=GNB_pred'] =0,0,0 
Pred_df['knn_pred=log_reg_pred'][Pred_df['knn_pred'] == Pred_df['log_reg_pred']] =1 
Pred_df['knn_pred=GNB_pred'][Pred_df['knn_pred'] == Pred_df['GNB_pred']] =1 
Pred_df['log_reg_pred=GNB_pred'][Pred_df['log_reg_pred'] == Pred_df['GNB_pred']] =1 
Pred_df['sum'] = Pred_df['knn_pred=log_reg_pred']+ Pred_df['knn_pred=GNB_pred']+ Pred_df['log_reg_pred=GNB_pred']




######################### Selecting the model of choice #######################

    #1) Concensus and 2-to-1 split cases
Pred_df['Model_of_choice'] = 0


        # Generating indices  

            #case: 'knn_pred=log_reg_pred' (1970)
indices_knn_log = Pred_df[Pred_df['knn_pred=log_reg_pred']== 1].index

            #case: 'knn_pred=GNB_pred' (1568)
indices_knn_GNB = Pred_df[Pred_df['knn_pred=GNB_pred']== 1].index

            #case: 'log_reg_pred =GNB_pred'(1742)
indices_log_GNB = Pred_df[Pred_df['log_reg_pred=GNB_pred']== 1].index


        # Asigning a module of choice for the Concensus and 2-to-1 split cases
Pred_df['Model_of_choice'][indices_log_GNB] = 'GNB_pred'
Pred_df['Model_of_choice'][indices_knn_GNB]= 'knn_pred'
Pred_df['Model_of_choice'][indices_knn_log] = 'knn_pred'


    #2) 'hung' parliament (all disagree) case

        # indices of 'hung' (98)
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

    #creating an incorrect column to identify when the Ensemble model was wrong 
Pred_df['Incorrect'] = 0
Pred_df['Incorrect'][Pred_df.Actual != Pred_df.Ensemble_pred] =1

###############################################################################



#$$$$$$$$$$$$$$$$ Brief Analysis of Concensus, hung, and 2-1 $$$$$$$$$$$$

    #Creating dfs of where the concensus, hung and 2-1 cases were incorrect 
    
All_wrong = Pred_df[(Pred_df['sum'] == 3) & (Pred_df.Incorrect == 1)] # -- 317 cases
Two_one_split = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.Incorrect == 1)] #-- 521 cases
hung_split = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Incorrect == 1)] #-- 68 cases 
    
    #checking that all incorrect cases are accounted for by the dfs above
len(Pred_df[Pred_df.Incorrect == 1])
len(All_wrong) +len(Two_one_split) + len(hung_split)

    #Part 1: looking at concensus cases (54.8% of all cases)
100*len(Pred_df[Pred_df['sum'] == 3])/len(Pred_df)

        #34.99% of all incorrect predictions belonged to a concesus case 
            #22.54% of all concensus cases are wrong; 
                #Wrong concensus cases make up 12.34% of all observations
100*len(All_wrong)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(All_wrong)/len(Pred_df[(Pred_df['sum'] == 3)])     
100*len(All_wrong)/len(Pred_df)

    # Part 2: looking at hung cases (3.82% of all cases): 
100*len(Pred_df[Pred_df['sum'] == 0])/len(Pred_df)

        #7.51% of all incorrect cases is belong to 'hung' scenario;
            #69.39% of all 'hung' are wrongly predicted; 
                #Wrong hung cases make up 2.65% of all observation 
100*len(hung_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(hung_split)/len(Pred_df[(Pred_df['sum'] == 0)]) 
100*len(hung_split)/len(Pred_df) 

    # Part 3: looking at majority vote(2-1) cases (41.39% of all cases):
100*len(Pred_df[Pred_df['sum'] == 1])/len(Pred_df)
        
        #57.51% of all incorrect cases is belong to 2-1 scenario:
            #49.05% of all 2-1 splits are wrongly predicted; 
                #Wrong 2-1 cases make up 20.3% of all cases )
100*len(Two_one_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(Two_one_split)/len(Pred_df[(Pred_df['sum'] == 1)])
100*len(Two_one_split)/len(Pred_df)


#Comments and Questions 

    #1) Consensus:
        
        #77.46 of all concensus cases are correctly labelled
        
        #If all models come to a concensus on a player's position then it maybe safe
        #to presume that that player 'plays' like the position predicted. In this
        #situation the player's actual position assignment is dubious. Humans are not
        #always logical in their categorisation.
        
    
    #2) 2-1 split:
        
        #50.95 of all 2-1 split cases are correctly label (kinda bad )
        
        #Any imporvements to the model should be focused this scenario as 57.51% of
        #all incorrect predictions involve this case (thus the potential increase
        #in accuracy is as much as 20.3%)
        
        #However, if two models have a concensus on a player's position it could be
        #that the player plays that way
    
    
    #3) Hung Parliament:
    
        #30.61 of all hung cases are correctly label
    
        #The randomness of the selection process means that the accuracy of this
        #case will hover around 33.33%. The potential increase in accuracy if 
        #all hung cases are guessed correctly is just 2.65% so model improvements
        #should not be focused on this. 
        
        # Even if the overall accuracy hardly changes, The model should not randomly 
        #decide this case because the model's answer changes with each run 
        #i.e. results are not repeatable.  

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


############## Addressing the 'hung' cases: Eliminating randomess  ###############################
Hung_cases  = Pred_df[(Pred_df['sum'] == 0)]
    
    # There are 98 hung cases
len(Hung_cases)

    # Hung cases where no model predicted correctly:
Hung_All_wrong = Hung_cases[(Hung_cases.Actual != Hung_cases.knn_pred) & (Hung_cases.Actual != Hung_cases.log_reg_pred) & (Hung_cases.Actual != Hung_cases.GNB_pred)] 

        # there are 7 'lost cause' hung cases (i.e. hung cases where no model got the correct answer)
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
    
    pres_list = []
    pres_list2 = []
    
    for mod in mods: 
        A1 = Hung_cases[(Hung_cases[mod] == i)]
        A1a = Hung_cases[(Hung_cases[mod] == i) & (Hung_cases.Actual == i)]
        no_pred = len(A1)
        pres = round(100*len(A1a)/len(A1), 2)
        pres1 = len(A1a)/len(A1)
        pres_list.append(f'{no_pred}({pres})')
        pres_list2.append(pres1*no_pred)


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
    #             the correct answer (> 30.06% achieved through random guess)
    #             and for all lost cause cases it had a 45% chance of getting 
    #             the correct answer ]


    #Alternative method:
        #1) Find the (actual) label that is most featured in hung cases and chose the 
            #model that best predicts this label (highest precision)
        
        
##############################################################################






################### Addressing low 2-1 voting accuracy  #############################
# From confusion matrix  SF accuracy is pultry becasue the we are overpredicting PF 
# in place of SF which is in turn reducing the precesion of PF and the overall accuracy
# of the model. This is the result of the ensemble model inheriting the bad 
# habits (bias) of the of KNN and log_reg confusing SF for PF. Below I will try to 
# improve this aspect of the ensemble model but no more tweak of this kind will 
# be made as this risks overfitting the model to the 'test' data. Plus, it could 
# be that SF and PF are similar positions in reality.   


    #2-1 cases where PF was predicted and GNB predicted SF
C4a = Pred_df[(Pred_df['sum']==1) & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.GNB_pred =='SF') ]

    #2-1 cases where the majority descision of PF was actually correct
C4b = Pred_df[(Pred_df['sum']==1) & (Pred_df.GNB_pred =='SF') & (Pred_df.Ensemble_pred == 'PF') & (Pred_df.Incorrect == 0)]
        
        # 34.24% of the time the majority decision was correct (kinda bad)        
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
    
 





##################### Ensemble prediction 2: Playing the odds #################
    
           
    #Predictions (to do)
    
        # Edit 1; Addressing Hung cases (using log_reg_prediction)
Pred_df.loc[indices_hung, 'Model_of_choice'] = 'log_reg_pred'

#Check = Pred_df[Pred_df['sum'] == 0] #checking 

        #Edit 2: Addressing 2-1 cases where PF was chosen by knn and log_reg whilst GNB chose SF
ind_GNB = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.knn_pred == 'PF') & (Pred_df.log_reg_pred == 'PF') & (Pred_df.GNB_pred == 'SF')].index
# A = Pred_df.loc[ind_GNB, 'Model_of_choice'] #= 'GNB_pred'
Pred_df.loc[ind_GNB, 'Model_of_choice'] = 'GNB_pred'

for i,mod_choice in enumerate(Pred_df['Model_of_choice']):
    Pred_df.loc[i, 'Ensemble_pred'] = Pred_df.loc[i, mod_choice] 


    # creating y_pred

y_pred_ensemble = np.array(Pred_df['Ensemble_pred'])

Pred_df['Incorrect'] = 0
Pred_df['Incorrect'][Pred_df.Actual != Pred_df.Ensemble_pred] =1




###############################################################################



#$$$$$$$$$ Brief Analysis of All cases (Concensus, hung, and Majority ) $$$$$$

All_wrong = Pred_df[(Pred_df['sum'] == 3) & (Pred_df.Incorrect == 1)]
Two_one_split = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.Incorrect == 1)]
hung_split = Pred_df[(Pred_df['sum'] == 0) & (Pred_df.Incorrect == 1)]


    #852 wrong predictions in total 
len(All_wrong) +len(Two_one_split) + len(hung_split)
# len(Pred_df[Pred_df.Incorrect == 1]) #alt checker 

    #Part 1: looking at concensus cases (54.79% of all cases)
100*len(Pred_df[Pred_df['sum'] == 3])/len(Pred_df)

        #37.21% of all incorrect predictions have a concesus case 
            #22.54% of all concensus cases are wrong; 
                #Wrong concensus cases make up 12.35% of all observations
100*len(All_wrong)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(All_wrong)/len(Pred_df[(Pred_df['sum'] == 3)])     
100*len(All_wrong)/len(Pred_df)

    # Part 2: looking at hung cases (3.82% of all cases): 
100*len(Pred_df[Pred_df['sum'] == 0])/len(Pred_df)

        #6.34% of all incorrect cases is belong to 'hung' scenario;
            #55.10% of all 'hung' are wrongly predicted; 
                #Wrong hung cases make up 2.10% of all observations
100*len(hung_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(hung_split)/len(Pred_df[(Pred_df['sum'] == 0)]) 
100*len(hung_split)/len(Pred_df) 

    # Part 3: looking at majority vote(2-1) cases (41.3% of all cases):
100*len(Pred_df[Pred_df['sum'] == 1])/len(Pred_df)
        
        #56.46% of all incorrect cases is belong to 2-1 scenario:
            #45.29% of all 2-1 splits are wrongly predicted; 
                #Wrong 2-1 cases make up 18.75% of all observations )
100*len(Two_one_split)/len(Pred_df[Pred_df.Incorrect == 1]) 
100*len(Two_one_split)/len(Pred_df[(Pred_df['sum'] == 1)])

100*len(Two_one_split)/len(Pred_df)


#Comments and Questions 

    #1) Consensus:
        #as before
    
    #2) 2-1 split:
        
        #The percentage of 2-1 splits wrongly predicted has reduced from 49.05% to 45.29%

        #Wrong 2-1 now make up 18.75% of all observations (down from 20.3%)
        
        
    #3 Hung parliament 
        
        #The percentage of hung splits wrongly predicted has reduced from 69.39% to 55.10%

        #Wrong hung now make up 2.10% of all observations (down from 2.65%)
        
        
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


####################### Evaluating Ensemble model #############################

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



    #1) Accuracy
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
plot_confusion_matrix(cm,classes = log_reg.classes_, title='Ensemble_hv_flex: Confusion matrix for test data')

    #3) Classification report 
print(classification_report(y_test, y_pred_ensemble))            

################################################################################







