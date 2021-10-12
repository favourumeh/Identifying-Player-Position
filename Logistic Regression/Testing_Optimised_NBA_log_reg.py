# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:46:48 2021

@author: favou
"""
import pickle
import pandas as pd
import numpy as np

#== Alter the following 5 variables with per-36 statistics ====================

FG_per = 0.5 # proportoion of Field goals made 
Three_attempts = 0.5 # three point shots attempted per 36 minutes
Three_per = 0.3 # proportion of three pointers made
Two_attempts = 12 # two point shots attemoted per 36mins 
Two_per = 0.5 # proportion of two point shots made
FTA = 10 # free throw attempts per 36 minutes
FT_per =0.7 # proportion of free throws made 
TRB = 12 #total rebounds per 36 minutes
AST = 2 #total assists per 36 minutes
STL = 1 #total steal per 36 minutes 
BLK = 0.2 #total blocks per 36 minutes
TOV = 1 # turnovers commited per 36 minutes 
PF = 2 # personal fouls per 36 minutes 


headings = ['FG_per', '3PA', '3P_per', '2PA', '2P_per', 'FTA', 'FT_per', 'TRB',
            'AST', 'STL', 'BLK', 'TOV', 'PF']
#==============================================================================




# load model 
with open('Optimal_log_reg_model.sav', 'rb') as pickle_file:
     log_reg_eval2 = pickle.load(pickle_file)

#loading df of unscaled per 36mins stats (for 10% untrained data)
df_unscaled = pd.read_csv('df_10.csv')[headings]

# coverting list of stats to dataframe of scaled stats  
L = [FG_per, Three_attempts,Three_per, Two_attempts, Two_per, FTA, FT_per, TRB, AST, STL, BLK, TOV, PF ]

#     #test uploaded model 
#L = list(df_unscaled[headings].loc[0,:])


def min_max_scaler_list(L, df):
    #Make a list or per-36min stats scaled 
    
    L_s = [] # holds the scaled stats
    for i in range(len(L)):
        
        if 'per' not in df.columns[i]:
            
            min_s = df.iloc[:,i].min()
            max_s = df.iloc[:,i].max()
            
            #in case the min or max statistic is written in section begining with line 10
            
            min_s = L[i] if L[i] < min_s else df.iloc[:,i].min()
            max_s = L[i] if L[i] > max_s else df.iloc[:,i].max()
                
            e_s = (L[i]-min_s)/(max_s-min_s)
        
        else: 
            e_s = L[i]
        
        L_s.append(e_s)
    
    return L_s

L_scaled = min_max_scaler_list(L, df_unscaled)

#Appending scaled list to dataframe
df_stats_scaled = pd.DataFrame(np.array(L_scaled).reshape(-1, len(L_scaled)), columns = headings)

#prediction 
position = log_reg_eval2.predict(df_stats_scaled)[0]
print(f'This player is probably a {position}')

