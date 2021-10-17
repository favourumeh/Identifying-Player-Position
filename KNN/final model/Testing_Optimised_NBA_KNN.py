# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:46:48 2021

@author: favou
"""
import pickle
import pandas as pd
import numpy as np

#== Alter the following 5 variables with per-36 statistics ====================

# Threes = 0 #three pointers made per 36 minutes
# Three_attempts = 0.5 #three pointers attempted per 36 minutes
# Three_per = Threes/Three_attempts # proportion of three pointers made
# ORB = 3 #offensive rebounds per 36 minutes 
# DRB = 0.5 #defensive rebounds per 36 minutes
# TRB = 26 #total rebounds per 36 minutes
# AST = 2 #total assists per 36 minutes
# STL = 1 #total steal per 36 minutes 
# BLK = 0.2 #total blocks per 36 minutes
# PF = 2 # personal fouls per 36 minutes 

headings = ['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']
#==============================================================================




# load model 
with open('Optimal_KNN_model_train.sav', 'rb') as pickle_file:
     knn5 = pickle.load(pickle_file)

#loading df of unscaled per 36mins stats (for 10% untrained data)
df_unscaled = pd.read_csv('df_10.csv')[headings]

# coverting list of stats to dataframe of scaled stats  
L = [Threes, Three_attempts, Three_per, ORB, DRB, TRB, AST, STL, BLK, PF]
#L =  [1.241379,  3.413793,  4.034483,  2.172414,  0.310345]

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
    #long way
# columns = headings

# series = pd.Series(L_scaled, index = columns)
# df_stats_scaled = pd.DataFrame()
# df_stats_scaled = df_stats_scaled.append(series, ignore_index=True)[headings]

    #short way
df_stats_scaled = pd.DataFrame(np.array(L_scaled).reshape(-1, len(L_scaled)), columns = headings)

#prediction 
position = knn5.predict(df_stats_scaled)[0]
print(f'This player is probably a {position}')

