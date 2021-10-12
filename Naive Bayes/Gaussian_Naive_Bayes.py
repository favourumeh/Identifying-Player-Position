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

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
import numpy as np
from statistics import mean
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

import scipy.stats as stats

from sklearn.model_selection import cross_val_score, cross_validate

#Generate data for analysis 
df = pd.read_csv('df_unscaled_with_pos.csv')
df.head()


#split 90-10
df_10 = df.sample(frac = 0.10)
df_90 = df.drop(df_10.index, axis =0) 


#Analysis 1: Hold out (train test split:split 90% into 78-22 train test)
X = df_90.drop('Pos', axis = 1)
y = df_90.Pos


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)

    #Checking distribution of independent varaibles 
X_train.hist() 
X_train.skew()
X_train.kurtosis()

    #checking the equivalent normal distributions for d_90
mean_std_df = pd.DataFrame() 
columns = ['features', 'PF', 'C', 'SG', 'SF', 'PG' ]


for column in df_90:
    if column != 'Pos':
        mean_std = []
        plt.figure()
        for position in df.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']           
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

#note: from the gaussian plots FT, and FTA should be removed as the data 
#       isn't spread across the different player positions



GNB = GaussianNB()
GNB.fit(X_train,y_train)

ma = GNB.score(X_test, y_test)
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


#Analysis 2: Cross Validation
ma_cv = cross_val_score(GNB, X, y, cv = 5, scoring = 'accuracy').mean()

print(f'For 5 fold cross validation: The mean mean accuracy = {ma_cv:.3f} ')


#Analysis 3:Tuning uisng Cross validation

    #Generating relative variance across classes(labels) for each feature
def relative_std(L1):
    #This calculates the relative standard deviation for numbers in list L1
    #(i.e. the average spread of data relative to the mean )
    rel_std = (sum([((i-mean(L1))/mean(L1))**2 for i in L1])/len(L1))**0.5
    return rel_std


A_st_dev = pd.DataFrame(columns = ['Variable', 'Relative Standard deviation', 'Categoric Mean(PF, C, SG, SF, PG)'])

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
X2 = df_90.drop(['Pos'], axis =1)

for c,i in enumerate(best_variables):
          
    GNB1 = GaussianNB()
    acc = cross_val_score(GNB1, X2, y2, cv = 5, scoring = 'accuracy').mean()    
    series = pd.Series([f'{list(X2.columns)}', acc], index = columns)
    Acc_df = Acc_df.append(series, ignore_index=True)
    
        
    X2 = X2.drop(i, axis = 1) 


       
#Evaluation of 10%

X3 = df_10.drop('Pos', axis =1)[['3P', '3PA', 'TRB', 'AST', 'BLK']]
y3 = df_10.Pos

X_train1, X_test1, y_train1, y_test1 = train_test_split(X3, y3, test_size=0.3, random_state=0)

GNB1 = GaussianNB().fit(X_train1, y_train1)

y_pred_test1 = GNB1.predict(X_test1)

ma3 = GNB1.score(X_train1,y_train1)
print(f'The mean accuracy of the training evaluation model(hold-out) in is : {ma3:.3f}')
ma3b = GNB1.score(X_test1,y_test1)
print(f'The mean accuracy of the testing evaluation model(hold-out) in is : {ma3b:.3f}')


ma4 = mean(cross_val_score(GNB1, X3, y3, scoring = 'accuracy'))
print(f'The mean cross-val accuracy of the evaluation model in is : {ma4:.3f}') # 0.646


cm3 = confusion_matrix(y_test1, y_pred_test1)

plt.figure()
plot_confusion_matrix(cm3,classes = GNB.classes_, title='GNB: Confusion matrix for test data')

print(classification_report(y_test1, y_pred_test1))

