# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:46:48 2021

@author: favou
"""
import pickle
import pandas as pd
import numpy as np

#== Alter the following variables with per-36 statistics ==============================================

Three_attempts = 1.5 # three-point shots attempted per 36 minutes
Three_per = 0.3 # proportion of three-poin shots made

Two_attempts = 10 # two-point shots attempted per 36mins
Two_per = 0.5 # proportion of two-point shots made

FTA = 7 # free throws attempted per 36 mins
FT_per = 0.8  # proportion of free throws made per 36 mins


ORB = 3 #offensive rebounds per 36 minutes 
DRB = 0.5 #defensive rebounds per 36 minutes

AST = 2 #total assists per 36 minutes
STL = 1 #total steal per 36 minutes 
BLK = 0.2 #total blocks per 36 minutes
TOV = 1 # turnovers commited per 36 minutes 
PF = 2 # personal fouls per 36 minutes 

headings = ['FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per', 'FTA', 'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
#========================================================================================================
Threes = Three_attempts*Three_per # three-point shots made per 36mins
Twos = Two_attempts*Two_per # two-point shots made per 36mins
FGA = Three_attempts + Two_attempts  # number of field goals attempted per 36mins  
FG_per = (Twos + Threes)/FGA # proportion of field goals made 
TRB = ORB + DRB #total rebounds per 36 minutes




# load model 
with open('Optimal_GNB_model_train.sav', 'rb') as pickle_file:
     GNB5 = pickle.load(pickle_file)


# coverting list of stats to dataframe of stats (needs to be in same order as headings variable) 
L = [FGA, FG_per, Threes, Three_attempts, Three_per, Twos, Two_attempts, Two_per, FTA, FT_per, ORB, DRB, TRB, AST, STL, BLK, TOV, PF]

# #     test uploaded model is correct
# df_unscaled = pd.read_csv('df_10.csv')
# L = list(df_unscaled[headings].loc[0,:])


#Appending  list to dataframe
df_stats = pd.DataFrame(np.array(L).reshape(-1, len(L)), columns = headings)

#Uncomment to look at your inputs 
# df_stats1 = pd.DataFrame(np.array(L).reshape(-1, len(L)), columns = headings).T
# df_stats1.columns = ['Player stats']
# print(df_stats1)

#prediction 
position = GNB5.predict(df_stats)[0]
print(f'This player is probably a {position}')

    #note: the order which you input your features to the model matters:
           # Make your features are the same order as the 'heading' variable 
           
           # Make sure the list variable, 'L', is also in the same order as the 
           # 'heading' variable as the df inputed(df_stats) in the model will take 
           # the values of this list as the columns values. 
