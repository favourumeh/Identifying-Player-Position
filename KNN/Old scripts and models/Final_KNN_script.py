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

#scaling function
def min_max_scaler(df):
    df5 = pd.DataFrame()
    for column in df:
        if 'per' not in column :
            df5[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
        
        if 'per' in column:
            df5[column] = df[column]
    return df5


# Importing raw data
df_100= pd.read_csv('Null handled with Pos.csv')

    # train_test split
X_train = pd.read_csv('X_train_correctly_scaled.csv')
X_test = pd.read_csv('X_test_correctly_scaled.csv')

y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

    # Evaluation 
df_n4_10 = pd.read_csv('df_n4_10.csv')
df_n4_10 = df_n4_10.iloc[:,1:]

df_n4_90 = df_100.drop(pd.read_csv('df_n4_10.csv').iloc[:,0].values, axis =0)


# Modeling KNN (k = 5)

    # Modelling 
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train.values.flatten())


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
y = df_n4_90.Pos
knn1 = KNeighborsClassifier(n_neighbors=5)

Accuracy_scores = cross_val_score(knn1, X_s, y, cv = 5, scoring = 'accuracy') # [0.59126298, 0.61894464, 0.60363479, 0.60752921, 0.61877975]
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

df_n4 = deepcopy(df_n4_90)
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
        for position in df_n4_90.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']   
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


X_s = min_max_scaler(df_n4_90.drop('Pos', axis =1))
X_tt1 = X_s[['3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF']]


knn_eval = KNeighborsClassifier(n_neighbors = 27)
knn_eval.fit(X_tt1, y.values.flatten())


y_pred = knn_eval.predict(X_tt1) 
y_pred_eval = knn_eval.predict(X_eval)

        # Evaluate df_10 set:
            #Accuracy
ac_train = accuracy_score(y_pred, y) # 0.62 accuracy
ac = accuracy_score(y_pred_eval, y_eval) # 0.62 accuracy

            #classifcation report
            
print(classification_report(y_eval, y_pred_eval))
            #confusion matrix 
cm2 = confusion_matrix(y_eval, y_pred_eval)


plt.figure()
plot_confusion_matrix(cm2, classes = knn.classes_, title='knn: Confusion matrix for evaluation data')



###################### Saving Model ###########################################

# #saving model 
# import pickle

# with open('Optimal_KNN_model.sav','wb') as f:
#       pickle.dump(knn_eval,f)
      
###############################################################################




################# Predicted Probabilities ####################################

#creating a dataframe of the predicted probabilities, the predictions and the actual labels
Pred_prob_df = pd.DataFrame(knn_eval.predict_proba(X_tt1), columns = list(knn_eval.classes_))
Pred_prob_df['Prediction'] = y_pred
Pred_prob_df['Actual'] = y.values.flatten()
Pred_prob_df['Incorrect'] = 0
Pred_prob_df['Incorrect'][Pred_prob_df.Prediction !=Pred_prob_df.Actual] =1

# #saving this dataframe for later insight 
# Pred_prob_df.to_csv('Pred_prob_KNN.csv', index = False)


# len(Pred_prob_df[Pred_prob_df.Incorrect == 1]) # number of incorrect predictions

#     #The worse precisions belong to SF and PF so lets see focus on these

# Incorrect_pred_df = Pred_prob_df[Pred_prob_df.Incorrect == 1]

###############################################################################
