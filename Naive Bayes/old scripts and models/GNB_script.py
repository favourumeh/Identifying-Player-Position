# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:16:13 2021

@author: favou
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from statistics import mean

import scipy.stats as stats

from copy import deepcopy
from sklearn.model_selection import cross_val_score

#scaling function
def min_max_scaler(df):
    df5 = pd.DataFrame()
    for column in df:
        if 'per' not in column :
            df5[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
        
        if 'per' in column:
            df5[column] = df[column]
    return df5


#importing data for analysis 

    #dataframes
df_100= pd.read_csv('Null handled with Pos.csv')
df_90 = df_100.drop(pd.read_csv('df_10.csv').iloc[:,0].values, axis =0)


    # train_test split
# X_train = pd.read_csv('X_train_correctly_scaled.csv')
# X_test = pd.read_csv('X_test_correctly_scaled.csv')

X_train = pd.read_csv('X_train_unscaled.csv')
X_test = pd.read_csv('X_test_unscaled.csv')

y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

    #train and test combined (90% of data)
a = pd.read_csv('X_train_unscaled.csv')[X_train.columns]
b = pd.read_csv('X_test_unscaled.csv')[X_train.columns]


    #Scaled 90% of data 
X_s = min_max_scaler(df_90.drop('Pos', axis = 1))
y = df_90.Pos
 
    # Evaluation dataset
df_10 = pd.read_csv('df_10.csv')
df_10 = df_10.iloc[:,1:]

    #checking the equivalent normal distributions for d_90
mean_std_df = pd.DataFrame() 
columns = ['features', 'PF', 'C', 'SG', 'SF', 'PG' ]


for column in df_90:
    if column != 'Pos':
        mean_std = []
        plt.figure()
        for position in df_90.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']           
            mu = df_90[column][df_90.Pos == position].mean() 
            sigma = df_90[column][df_90.Pos == position].std()
            
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label = f'{position}') 
            plt.title(f'The equivalent gaussian (normal) distribution of {column} across different player position ')
            plt.legend()
            plt.xlabel(f'{column}')
            plt.ylabel('Probability(pdf')
            mean_std.append([mu, sigma])
            
        mean_std.insert(0,column)
        series = pd.Series(mean_std, index = columns)
        mean_std_df = mean_std_df.append(series, ignore_index = True)     

mean_std_df = mean_std_df[['features','C', 'PF', 'PG', 'SF', 'SG']]

#note: from the gaussian plots FG, FT, and FTA should be removed as the data 
#       isn't spread across the different player positions


# Modeling log_reg

    # Modelling 
GNB = GaussianNB()
GNB.fit(X_train,y_train.values.flatten())

        # Evaluate test dataset on metrics (Accuracy)
ma_tr = GNB.score(X_train, y_train) # accuracy = 0.619 (Unscaled data)       
ma = GNB.score(X_test, y_test) # accuracy = 0.543(scaled data), 0.608(unscaled data)
print(f'The mean accuracy of the hold out model in is : {ma:.3f}')

y_pred = GNB.predict(X_train)
y_pred_test = GNB.predict(X_test)

    #classification report 
print(classification_report(y_test, y_pred_test))

    #Confusion matrix
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
plot_confusion_matrix(cm,classes = GNB.classes_, title='GNB: Confusion matrix for trained data')
plt.figure()
plot_confusion_matrix(cm1, classes = GNB.classes_, title='GNB: Confusion matrix for test data')


#Analysis 2: Cross Validation: to guage how typical the test accuracy is

X_100 = df_100.drop('Pos', axis=1)[X_train.columns]
y_100 = df_100.Pos
#X_100 = min_max_scaler(X_100)

GNB1 = GaussianNB()
ma_cv = cross_val_score(GNB1, X_100, y_100.values.flatten(), cv = 5, scoring = 'accuracy').mean()

print(f'For 5 fold cross validation: The mean mean accuracy = {ma_cv:.3f} ') #0.604, 0.603(unscaled)  


#Analysis 3:Tuning uisng Cross validation

    #Generating relative variance across classes(labels) for each feature
    
def relative_std(L1):
    #This calculates the relative standard deviation for numbers in list L1
    #(i.e. the average spread of data relative to the mean )
    rel_std = (sum([((i-mean(L1))/mean(L1))**2 for i in L1])/len(L1))**0.5
    return rel_std


A_st_dev = pd.DataFrame(columns = ['Variable', 'Relative Standard deviation', 'Categoric Mean(PF, C, SG, SF, PG)'])

df_90 = df_100.drop(pd.read_csv('df_10.csv').iloc[:,0].values, axis =0)

for i in df_90:
    
    if i not in ('Pos'):  
        mean_stat_list  =[]
        for position in ['PF', 'C', 'SG', 'SF', 'PG']:
            index = df_90.loc[:, i][df_90.Pos == position ].index
            mean_stat = df_90.loc[index, i].mean()
            #print(f'For position {position} the mean {i} =  {mean_stat:.3f}')
            
            mean_stat_list.append(mean_stat)
      

        A_st_dev = A_st_dev.append({'Variable': i, 'Relative Standard deviation': f'{relative_std(mean_stat_list):.3f}', 'Categoric Mean(PF, C, SG, SF, PG)': f'{[round(i, 3) for i in mean_stat_list]}'}, ignore_index = True) 



    #Progressively reducing noisiest data 
best_variables = A_st_dev.sort_values(by = 'Relative Standard deviation', axis =0, ascending = True)['Variable'].values
columns = ['Variable combo', 'Accuracy']
Acc_df  = pd.DataFrame(columns = columns)

y2 = df_90['Pos']
X2 =df_90.drop(['Pos'], axis =1)
#X2 = min_max_scaler(df_90.drop(['Pos'], axis =1))

for c,i in enumerate(best_variables):
          
    GNB2 = GaussianNB()
    acc = cross_val_score(GNB2, X2, y2, cv = 5, scoring = 'accuracy').mean()    
    series = pd.Series([f'{list(X2.columns)}', acc], index = columns)
    Acc_df = Acc_df.append(series, ignore_index=True)
    
        
    X2 = X2.drop(i, axis = 1) 


       
#Evaluation of 10%

X_eval = df_10.drop('Pos', axis = 1)[['FG_per', '3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
#_eval = min_max_scaler(X_eval)

y_eval = df_10.Pos

X_90 = df_90.drop('Pos', axis =1)[['FG_per', '3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
#_90 = min_max_scaler(X_90)

y_90 = df_90.Pos
GNB3 = GaussianNB().fit(X_90, y_90)

y_pred = GNB3.predict(X_90)
y_pred_eval = GNB3.predict(X_eval)

ma3 = GNB3.score(X_90,y_90)# 0.615(unscaled)
print(f'The mean accuracy of the training evaluation model(hold-out) in is : {ma3:.3f}') 
ma3b = GNB3.score(X_eval,y_eval)# 0.519, 0.643
print(f'The mean accuracy of the testing evaluation model(hold-out) in is : {ma3b:.3f}')



  


cm3 = confusion_matrix(y_eval, y_pred_eval)

plt.figure()
plot_confusion_matrix(cm3,classes = GNB.classes_, title='GNB: Confusion matrix for evaluation data')

print(classification_report(y_eval, y_pred_eval))


###################### Saving Model ###########################################

# #saving model 
# import pickle

# with open('Optimal_GNB_model.sav','wb') as f:
#       pickle.dump(GNB3,f)
      
###############################################################################






################# Predicted Probabilities ####################################

#creating a dataframe of the predicted probabilities, the predictions and the actual labels
Pred_prob_df = pd.DataFrame(GNB3.predict_proba(X_90), columns = list(GNB3.classes_))
Pred_prob_df['Prediction'] = y_pred
Pred_prob_df['Actual'] = y_90.values.flatten()
Pred_prob_df['Incorrect'] = 0
Pred_prob_df['Incorrect'][Pred_prob_df.Prediction !=Pred_prob_df.Actual] =1

# #saving this dataframe for later insight 
# Pred_prob_df.to_csv('Pred_prob_GNB.csv', index = False)


# len(Pred_prob_df[Pred_prob_df.Incorrect == 1]) # number of incorrect predictions

#     #The worse precisions belong to SF and PF so lets see focus on these

# Incorrect_pred_df = Pred_prob_df[Pred_prob_df.Incorrect == 1]

###############################################################################

