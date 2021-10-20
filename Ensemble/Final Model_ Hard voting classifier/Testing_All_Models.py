# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:26:54 2021

@author: favou
"""
from Ensemble_Hard_Voting_function import model_function
import pandas as pd

# Chose a model 
model = 'E_hv_flex' # (options: 'KNN', 'GNB', 'log_reg', 'E_hv1', 'E_hv2' or 'E_hv_flex') 

# Predicting one player or multiple players? 
multiple = True #(options: True or False)

# Look at accuracy, classifcation report and confusion matrix
Eval1 = True # (not available for single player predictors)



################# Predicting the position of a single player #####################

#== Alter the following variables with per-36 statistics ======================
Three_attempts = 1.5 # three-point shots attempted per 36 minutes
Three_per = 0.3 # proportion of three-point shots made
Two_attempts = 10 # two-point shots attempted per 36mins
Two_per = 0.5 # proportion of two-point shots made
FTA = 7 # free throws attempted per 36 mins
FT_per = 0.8  # proportion of free throws made per 36 mins
ORB = 3 # offensive rebounds per 36 minutes 
DRB = 0.5 # defensive rebounds per 36 minutes
AST = 8 # total assists per 36 minutes
STL = 1 # total steal per 36 minutes 
BLK = 0.2 #total blocks per 36 minutes
TOV = 1 # turnovers commited per 36 minutes 
PF = 2 # personal fouls per 36 minutes 
#==============================================================================

#Variables derived from the above variables
Threes = Three_attempts*Three_per # three-point shots made per 36mins
Twos = Two_attempts*Two_per # two-point shots made per 36mins
FGA = Three_attempts + Two_attempts  # number of field goals attempted per 36mins  
FG_per = (Twos + Threes)/FGA # proportion of field goals made 
TRB = ORB + DRB #total rebounds per 36 minutes

# Varibales not used by any models
FG, FT, PTS, eFG_per = 0,0,0,0 # (we still need these so the df is complete )

Pos = 'C' #just a placeholder postion (so the df is complete)

L1 = [Pos, FG, FGA, FG_per, Threes, Three_attempts, Three_per, Twos, Two_attempts,
       Two_per, eFG_per, FT, FTA, FT_per, ORB, DRB, TRB, AST,
       STL, BLK, TOV, PF, PTS]

###############################################################################



################# Predicting position of multiple players #####################
# Data File 
f = 'df_10.csv' # the file containing tabular per 36min data of multiple players 

#note: you can also try f = 'df_test.csv' to replicate results for testing data
        #However, you should only really try this on the non-ensemble models as the
        # ensemble models have been 'trained' with information from the test data set
###############################################################################


















if multiple == True:
    y_pred = model_function (multiple, model, data= f, L =L1, Eval = Eval1) # the y_predicted by the model 
else:
    y_pred = model_function(multiple, model, L =L1, Eval = Eval1)[-1] # the y_predicted by the model 
    print(f'This player is probably a {y_pred}')