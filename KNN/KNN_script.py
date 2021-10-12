# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:00:22 2021

@author: favou
"""
#Importing libraries
import pandas as pd 

from copy import deepcopy
from statistics import pstdev, mean 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

import math
import numpy as np
import scipy.stats as stats

# Importing raw data
df = pd.read_excel('1990_2021_NBA_ppg_stats.xlsx')

# Per game -----> per 36mins 
df1 = df[df["MP"]>10] # -- 12841 rows
df1.index = range(len(df1.index))

df1.drop(['Player','Age', 'Tm', 'G', 'GS', 'Year'],axis = 1, inplace = True)

df_n1 =  deepcopy(df1)

df_n2 = pd.DataFrame()
df_n2['Pos'] = df_n1.Pos
df_n2['3P_per'] = df_n1['3P_per'] 
df_n2['2P_per'] = df_n1['2P_per'] 
df_n2['FT_per'] = df_n1['FT_per'] 
df_n2['eFG_per'] = df_n1['eFG_per'] 


    # Per_game statistics ----> per 36 statistsics 
for c, column in enumerate(df_n1):
    if column not in ('Pos','3P_per', '2P_per', 'FT_per', 'eFG_per'):    
        df_n2[column] = (36*df_n1[column])/df_n1['MP']
        
df_n3 = df_n2.drop('MP', axis = 1)

df_n3 = df_n3[['Pos', 'FG', 'FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA',
       '2P_per', 'eFG_per', 'FT', 'FTA', 'FT_per', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PTS']]

df_n3.isnull().sum()
df_n4 = deepcopy(df_n3)
        
# Addressing Null values (categoric means)

    #Making the null shot percentage and zero shot attempt = mean 
        #(note: it doesn't matter that '3P', '2P', 'FT are altered in the for loop below we change these in the next'for loop' )

for i in ['3P_per', '2P_per', 'FT_per', '3PA', '2PA', 'FTA', '3P', '2P', 'FT']:
    
    for position in ['PF', 'C', 'SG', 'SF', 'PG']:
        if 'per' in i:
            index = df_n4.loc[:, i][(df_n4.Pos == position) & (df_n4[i].isnull())].index
            mean_stat = df_n4[i][(df_n4[i].notnull()) & (df_n4.Pos == position) ].mean()
            df_n4.loc[index, i] = mean_stat
            print(f'For position {position} the mean {i} =  {mean_stat:.3f}')

        if 'per' not in i:
            index = df_n4.loc[:, i][(df_n4.Pos == position ) & (df_n4[i]==0)].index
            index1 = df_n4.loc[:, i][(df_n4.Pos == position )].index
            mean_stat = df_n4.loc[index1, i].mean()
            df_n4.loc[index, i] = mean_stat
            print(f'For position {position} the mean {i} =  {mean_stat:.3f}')
    
    print("")
    
    # Making the shots made = shot_per *shot_attemps 
for i in ['3P', '2P', 'FT']:
    df_n4[i] = df_n4[(i+'A')]*df_n4[(i + '_per')]
    
    
df_n4.isnull().sum()

#df_n4.to_csv('nulls handled .csv', index = False)



# Feature scalling (minmaxscaler) 

def min_max_scaler(df):
    df5 = pd.DataFrame()
    for column in df:
        if 'per' not in column :
            df5[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
        
        if 'per' in column:
            df5[column] = df[column]
    return df5


 
df_n4_10 =df_n4.sample(frac = 0.10)
#df_n4_10.to_csv('df_n4_10.csv')

df_n4_90 = df_n4.drop(df_n4_10.index, axis = 0)   

# Modeling KNN (k = 5)
X = df_n4_90.drop('Pos', axis =1 )
y = df_n4_90.Pos


    #train-test split (77.8-22.2) (saved in current repository )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.222, random_state = 0)  

# X_train.to_csv('X_train_unscaled.csv', index = False)
# X_test.to_csv('X_test_unscaled.csv', index = False)

    #normalising the train dataset
X_train = min_max_scaler(X_train)
    #normalising the test dataset 
X_test = min_max_scaler(X_test)

# X_train.to_csv('X_train_correctly_scaled.csv', index = False)
# X_test.to_csv('X_test_correctly_scaled.csv', index = False)
# y_train.to_csv('y_train.csv', index = False)
# y_test.to_csv('y_test.csv', index = False)
 
    # Modelling 
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)


        # Evaluate test dataset on metrics (Accuracy)
from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

ac_train = accuracy_score(y_pred, y_train) # 0.7624 accuracy
ac = accuracy_score(y_pred_test, y_test) # 0.577 accuracy



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
plot_confusion_matrix(cm,classes = knn.classes_, title='knn: Confusion matrix for trained data')
plt.figure()
plot_confusion_matrix(cm1, classes = knn.classes_, title='knn: Confusion matrix for test data')

         
        # classification report 
print(classification_report(y_train, y_pred)) #training data 

print(classification_report(y_test, y_pred_test)) # testing data

        # Cross validation
from sklearn.model_selection import cross_val_score

X_s = min_max_scaler(df_n4_90.drop('Pos', axis =1))

Accuracy_scores = cross_val_score(knn, X_s, y, cv = 5, scoring = 'accuracy') # [0.59126298, 0.61894464, 0.60363479, 0.60752921, 0.61877975]
mean_Accuracy_score = Accuracy_scores.mean() # 0.608 slighly higher than accuracy of test set 0.577
print(Accuracy_scores)

    


# Tunning Model 

    #1) Removing Noise: Adjusting independent variables

def relative_std(L1):
    #This calculates the relative standard deviation for numbers in list L1
    #(i.e. the average spread of data relative to the mean )
    rel_std = (sum([((i-mean(L1))/mean(L1))**2 for i in L1])/len(L1))**0.5
    return rel_std

        #Generating a dataframe of the relative stadard diviation of the mean variable values 
        #for each outcome(PF, C, ...) for a given variable (see ''st_dev'' if confused)

st_dev = pd.DataFrame(columns = ['Variable', 'Relative Standard deviation', 'Categoric Mean(PF, C, SG, SF, PG)'])

for i in df_n4:
    
    if i not in ('Pos'):  
        mean_stat_list  =[]
        for position in ['PF', 'C', 'SG', 'SF', 'PG']:
                index = df_n4.loc[:, i][df_n4.Pos == position ].index
                mean_stat = df_n4.loc[index, i].mean()
                #print(f'For position {position} the mean {i} =  {mean_stat:.3f}')
                
                mean_stat_list.append(mean_stat)
      

        st_dev = st_dev.append({'Variable': i, 'Relative Standard deviation': f'{relative_std(mean_stat_list):.3f}', 'Categoric Mean(PF, C, SG, SF, PG)': f'{[round(i, 3) for i in mean_stat_list]}'}, ignore_index = True) 




    
            # Data visualiser: looking at the spread 
for column in df_n4:
    if column != 'Pos':
        plt.figure()
        for position in df.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']   
            mu = df_n4[column][df_n4.Pos == position].mean() 
            sigma = df_n4[column][df_n4.Pos == position].std()
            
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label = f'{position}')
            
            # g = df_n4[column][df.Pos == position].hist(label= position)
            # #g = sns.kdeplot(data=df_n4[column][df.Pos == position],label= position)
            # plt.title(f'{column}')
            
            plt.title(f'The equivalent gaussian (normal) distribution of {column} across different player position ')
            plt.legend()
            plt.xlabel(f'{column}')
            plt.ylabel('Probability(pdf)')
            

            # Dropping variables
best_variables = st_dev.sort_values(by = 'Relative Standard deviation', axis =0, ascending = True)['Variable'].values

#                 #combining training and testing set 
# X_tt = pd.concat([X_train, X_test], axis =0)# 0=union, 1 = join
# y_tt = pd.concat([y_train, y_test], axis =0)   

                #Using 90% of data 
X_s = min_max_scaler(df_n4_90.drop('Pos', axis =1))

                #Progressively dropping variables
                
columns = ['Variable combo', 'Accuracy']
Acc_df  = pd.DataFrame(columns = columns)

for c,i in enumerate(best_variables):
    
            
    knn2 = KNeighborsClassifier(n_neighbors = 5)
    acc = cross_val_score(knn2, X_s, y, cv = 5, scoring = 'accuracy').mean()    
    series = pd.Series([f'{list(X_s.columns)}', acc], index = columns)
    Acc_df = Acc_df.append(series, ignore_index=True)
    
        
    X_s = X_s.drop(i, axis = 1)
    
            # From analysis the least noise combo of features is: 
                # ['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']
                # yeilding a mean cross_val accuracy of 0.6547

    #2) Adjusting k
#X_tt1 = pd.concat([X_train, X_test], axis =0)# 0=union, 1 = join
X_s = min_max_scaler(df_n4_90.drop('Pos', axis =1))

X_tt1 = X_s[['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']]

k_range = range(1, 40)
mean_accuracy_scores3 = []
for k in k_range:
    knn3 = KNeighborsClassifier(n_neighbors = k)
    mean_accuracy_score3 = cross_val_score(knn3, X_tt1, y, cv = 5, scoring = 'accuracy').mean()
    mean_accuracy_scores3.append(mean_accuracy_score3)

K_df = pd.DataFrame(zip(k_range, mean_accuracy_scores3 ), columns = ['K_value', 'Mean_Accuracy'])

plt.figure()
plt.plot(k_range, mean_accuracy_scores3)
plt.xlim(5,40)
plt.xticks(ticks = range(5, 40, 2))

            # From Analysis optimal k = 27 (yields a mean cross_val accuracy of 0.6811)

    #3) Evaluating model 
X_eval = df_n4_10.drop('Pos', axis = 1)[['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']]

X_eval = min_max_scaler(X_eval)
y_eval = df_n4_10.Pos


knn_eval = KNeighborsClassifier(n_neighbors = 27, weights= 'distance')
knn_eval.fit(X_tt1, y)


y_pred_eval = knn_eval.predict(X_eval)

        # Evaluate df_10 set:
            #Accuracy
ac = accuracy_score(y_pred_eval, y_eval) # 0.61 accuracy
            #classifcation report
print(classification_report(y_eval, y_pred_eval))
            #confusion matrix 
cm2 = confusion_matrix(y_eval, y_pred_eval)


plt.figure()
plot_confusion_matrix(cm2, classes = knn.classes_, title='knn: Confusion matrix for evaluation data')



###################### Saving Model ###########################################

#saving model 
import pickle

with open('Optimal_KNN_model.sav','wb') as f:
      pickle.dump(knn_eval,f)
      
###############################################################################