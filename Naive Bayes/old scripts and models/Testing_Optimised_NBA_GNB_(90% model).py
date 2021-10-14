# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:46:48 2021

@author: favou
"""
import pickle
import pandas as pd
import numpy as np

#== Alter the following variables with per-36 statistics ====================

FG_per = 0.5 # proportoion of Field goals made 
Threes = 1 # three point shots made per 36mins
Three_attempts = 0.5 # three point shots attempted per 36 minutes
Three_per = Threes/Three_attempts # proportion of three pointers made
ORB = 3 #offensive rebounds per 36 minutes 
DRB = 0.5 #defensive rebounds per 36 minutes
TRB = 16 #total rebounds per 36 minutes
AST = 2 #total assists per 36 minutes
STL = 1 #total steal per 36 minutes 
BLK = 0.2 #total blocks per 36 minutes
TOV = 1 # turnovers commited per 36 minutes 
PF = 2 # personal fouls per 36 minutes 


headings = ['FG_per', '3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
#==============================================================================


# load model 
with open('Optimal_GNB_model.sav', 'rb') as pickle_file:
     GNB5 = pickle.load(pickle_file)

#loading df of unscaled per 36mins stats (for 10% untrained data)
df_unscaled = pd.read_csv('df_10.csv')[headings]

# coverting list of stats to dataframe of scaled stats  
L = [FG_per, Threes, Three_attempts,Three_per, ORB, DRB, TRB, AST, STL, BLK, TOV, PF]
# #     #test uploaded model 
# L = list(df_unscaled[headings].loc[0,:])


#Appending  list to dataframe
df_stats = pd.DataFrame(np.array(L).reshape(-1, len(L)), columns = headings)

#prediction 
position = GNB5.predict(df_stats)[0]
print(f'This player is probably a {position}')

