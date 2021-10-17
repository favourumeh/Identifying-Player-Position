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

from essential_functions import min_max_scaler, plot_confusion_matrix
  

def model_function (Multi, mod, data= None, L = None, Eval = False):
    
    
    # This function takes a DataFrame ('data') of a basketball player's game statistics 
    # per 36mins from Basketball-Reference.com and identifies the position of a player 
    # using one of 6 models (i.e 'mod' = 'KNN', 'GNB', 'log_reg', 'E_hv1', 'E_hv2' or 'E_hv_flex'). 

#note: 'data' is the name of the file containing the 

    # E_hv1, E_hv2 and E_hv_flex are ensemble models that use KNN, GNB and log_reg
    # to identify player positions. 
    
    # E_hv1 simply uses a hard-voting to decide a player's position. In the cases
    # where we have a 'hung parliament' (i.e. the model cannot establish a 
    # majority vote) the player postion is decided by choosing a random model 
    # to be the decider
    
    # E_hv2 also uses a hard-voting to decide a player's position. However, 
    # if we have a 'hung parliament' the player's postion is NOT decided by 
    # by random and instead we chose the prediction from the model with the 
    # highest precision based on the test data (i.e. log_reg) to be the decider
    
    # E_hv_flex is a flexible hard-voting system. For the most part the majority 
    # vote is used, but in the situations where KN and log_reg predict 'PF' and GNB 
    # predicts 'SF' GNB's prediction is used. 

    
    # You have the option to evaluate a model's performace on the data by setting
    # 'Eval' = True
    
    
    
    
    # Loading Non-ensemble Models 
    
        #KNN
    with open('Optimal_KNN_model_train.sav', 'rb') as pickle_file:
         knn = pickle.load(pickle_file)
    
        #log_reg
    with open('Optimal_log_reg_model_train.sav', 'rb') as pickle_file:
         log_reg = pickle.load(pickle_file)
         
        #GNB(takes unscaled data)
    with open('Optimal_GNB_model_train.sav', 'rb') as pickle_file:
         GNB = pickle.load(pickle_file)
        
        
        
    # Import unscaled model (per 36 minute stats) 
    
    if Multi == True:
        df_10 = pd.read_csv(data).iloc[:,1:]
    else:
        df_10_1 = pd.read_csv('df_10.csv').iloc[:,1:]
        
        df_10 = df_10_1.append( pd.DataFrame(np.array(L).reshape(-1, len(L)), columns = df_10_1.columns), ignore_index = True )

        
    y_test = df_10.Pos
    X_test = df_10.drop('Pos', axis =1)
    
    #convert X_test columns to float
    for column in X_test:
        X_test[column] = X_test[column].astype(float)
        
    #Scale model     
    X_test_s = min_max_scaler(X_test)
    
    
    # Features taken by each non-Ensemble model 
        #for KNN
    headings_knn = ['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']
    
        #for log_reg
    headings_log_reg = ['FG_per', '3PA', '3P_per', '2PA', '2P_per', 'FTA', 'FT_per', 'TRB',
                'AST', 'STL', 'BLK', 'TOV', 'PF']  
    
        #for naive bayes 
    headings_GNB = ['FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per', 'FTA',
                'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    
    
    # Non-Ensemble model Predictions
    y_knn = knn.predict(X_test_s[headings_knn])
    y_log_reg = log_reg.predict(X_test_s[headings_log_reg])
    y_GNB = GNB.predict(X_test[headings_GNB])
    
    
##################### Choosing a non-ensemble model to predict ################
    
    if mod == 'KNN':
        y_pred = y_knn
    
    
    if mod == 'log_reg':
        y_pred = y_log_reg
    
    
    if mod == 'GNB':
        y_pred = y_GNB
    
###############################################################################



    
############################## Ensemble process ##############################

    if mod in ('E_hv1', 'E_hv2', 'E_hv_flex'):
        
        #Creating Column of Actual label, Model Predecition and Voting aggreements
        Pred_df = pd.DataFrame(zip(y_test.values.flatten(), y_knn, y_log_reg, y_GNB), columns = ['Actual','knn_pred', 'log_reg_pred', 'GNB_pred'])
        Pred_df['knn_pred=log_reg_pred'], Pred_df['knn_pred=GNB_pred'], Pred_df['log_reg_pred=GNB_pred'] =0,0,0 
        
        D1 = Pred_df['knn_pred=log_reg_pred'][Pred_df['knn_pred'] == Pred_df['log_reg_pred']].index 
        D2 = Pred_df['knn_pred=GNB_pred'][Pred_df['knn_pred'] == Pred_df['GNB_pred']].index 
        D3 = Pred_df['log_reg_pred=GNB_pred'][Pred_df['log_reg_pred'] == Pred_df['GNB_pred']].index

        Pred_df.loc[D1, 'knn_pred=log_reg_pred'] = 1
        Pred_df.loc[D2, 'knn_pred=GNB_pred'] =1 
        Pred_df.loc[D3, 'log_reg_pred=GNB_pred'] =1 

        Pred_df['sum'] = Pred_df['knn_pred=log_reg_pred']+ Pred_df['knn_pred=GNB_pred']+ Pred_df['log_reg_pred=GNB_pred']
        Pred_df['Model_of_choice'] = 0
        Pred_df['Ensemble_pred'] = 0        
        
        
        #Selecting the model of choice based on majority voting 
        
        #1) Concensus and 2-to-1 split cases        
        
            # Generating indices  
        
                #case: 'knn_pred=log_reg_pred' (1970)
        indices_knn_log = Pred_df[Pred_df['knn_pred=log_reg_pred']== 1].index
        
                #case: 'knn_pred=GNB_pred' (1568)
        indices_knn_GNB = Pred_df[Pred_df['knn_pred=GNB_pred']== 1].index
        
                #case: 'log_reg_pred =GNB_pred'(1742)
        indices_log_GNB = Pred_df[Pred_df['log_reg_pred=GNB_pred']== 1].index
        
        
            # Asigning a module of choice for the Concensus and 2-to-1 split cases
        Pred_df.loc[indices_log_GNB, 'Model_of_choice']= 'GNB_pred'
        Pred_df.loc[indices_knn_GNB, 'Model_of_choice']= 'knn_pred'
        Pred_df.loc[indices_knn_log, 'Model_of_choice']= 'knn_pred'
        
        
        #2) 'hung' parliament (all disagree) case
        
                # indices of 'hung' (98)
        indices_hung = Pred_df[Pred_df['sum'] == 0].index
        
     
        
             #2A) Ensemble: Random Hard Voting 1.0 ##################
        if mod == 'E_hv1':
            
                    # import random module
            import random
            
            
            options = ['knn_pred', 'log_reg_pred', 'GNB_pred']
            rand_opt = np.array([random.choice(options) for i in range(len(indices_hung))]).reshape(-1,1) # randomly choses from 'options' list
            Pred_df.loc[indices_hung, 'Model_of_choice'] = rand_opt.flatten()
            
       
        
            #2B) Ensemble: Hard voting + Playing the odds #################                    
        if mod == 'E_hv2':
            # Edit 1; Addressing Hung cases (using log_reg_prediction)
            Pred_df.loc[indices_hung, 'Model_of_choice'] = 'log_reg_pred'
        
        
            #2C) Ensemble: Flexible Hard voting + Playing the odds #################                    
        if mod == 'E_hv_flex':       
            #Edit 2: Addressing 2-1 cases where PF was chosen by knn and log_reg whilst GNB chose SF
            Pred_df.loc[indices_hung, 'Model_of_choice'] = 'log_reg_pred'
            ind_GNB = Pred_df[(Pred_df['sum'] == 1) & (Pred_df.knn_pred == 'PF') & (Pred_df.log_reg_pred == 'PF') & (Pred_df.GNB_pred == 'SF')].index
            Pred_df.loc[ind_GNB, 'Model_of_choice'] = 'GNB_pred'
        
        
        # creating y_pred
        for i,mod_choice in enumerate(Pred_df['Model_of_choice']):
            Pred_df.loc[i, 'Ensemble_pred'] = Pred_df.loc[i, mod_choice] 
        
        
        y_pred = np.array(Pred_df['Ensemble_pred'])
        
        ##creating an incorrect column to identify when the Ensemble model was wrong 
        # Pred_df['Incorrect'] = 0
        # Pred_df['Incorrect'][Pred_df.Actual != Pred_df.Ensemble_pred] =1
    
    
    
    
###############################################################################
    
    
    

    
    
    
       
####################### Evaluating a Model ####################################

    if Eval == True & Multi == True:
        
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        
            #1) Accuracy
        ac_test = accuracy_score(y_test, y_pred) # 0.647 accuracy  #0.668
        print(f'The accuracy of the model {mod} = {ac_test:.2f}')
                
            #2) Confusion Matrix 
        cm = confusion_matrix(y_test, y_pred) 
        plt.figure()
        plot_confusion_matrix(cm,classes = log_reg.classes_, title= f'{mod}: Confusion matrix for Evaluation data')
        
            #3) Classification report 
        print("")
        print("The Classification Report: ")
        print(classification_report(y_test, y_pred))            
        
###############################################################################



    return y_pred


