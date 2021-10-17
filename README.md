# Identifying-Player-Position
 

## Project overview 
 - Created a tool than enables an NBA scout to identify an NBA player's position based on their per 36minutes statitics using 6 classifcation models. 
 - Data for this project was taken from basketballreference.com using the scraper developed from an earlier [Data Pipeline](https://github.com/favourumeh/DATA-PIPELINE) project 
 - This project uses three different classifcation algorithms: Naive Bayes, Logistic Regression and K Nearest Neighbours
 - In total six models were created:
    - three models used the individual algorithms mentioned above. They are called: 'KNN', 'log_reg' and 'GNB'
    - three more models were created using using an ensemble appraoch. These ensemble models use hard voting with some minor tweaks based on anlysisis conducted on confusion matrix. They are called: 'E_hv1', 'E_hv2' and 'E_hv_flex' 
 -  All model were evaluated by comparing each other agianst three classifcation metrics : precision, recall and mean accuracy. Their confusion matrices were also analysed to gauge how well each model performed for different labels (/classes). 
 -  This is a summary of the findings for each model:
                                               
                                               [INSERT TABLE]
                                              
 -  Feel free to try out the tool created by cloning the repository and opening the file: 'Testing_All_Models.py'. Follow the flowchart below for further guidance.  
 
 ## Python version and packages 
Python Version: **3.8.3**

Packages: pandas, numpy, sklearn, matplotlib, seaborn, pickle


 ## Project context
 Before diving into the project details it is important to understand some project context. **See file: 'Data_explained.docx'** for greater context of the tabular data 
 
 
 ## Data Splitting 
 The data used for this report was split into three sets: **1) Training set(70%); 2) Testing set(20%) and 3) Evaluation set (10%)**. 
 
 The non-ensemble models were trained and optimised (i.e. using hyper parameter tuning or feature selection) using only the training set. As a result of this the non-ensemble models could be evaluated based on the testing and evaluations sets. 
 
  Making improvements to the ensemble model required the use of data different to the training set because KNN involves telling the model the correct prediction and the non-ensemble models used were already optimised for the training set. The testing set was, thus, used to 'train' the ensemble model (see ensemble section for more information). 
  
  All models were evaluated using the evaluation set as this data was totaly unseen by both sets of models. 
 
 
  
 ## Feature Selection 
 - feature scalling
 -K-Nearest Neighbours
 -Logistic Regression
 -Gaussian Naive Bayes 
 
 ## Hyperparameter tunning 
  -K-Nearest Neighbours
 -Logistic Regression
 -Gaussian Naive Bayes 
 
 ## Evaluating KNN, log_reg and GNB
 
 ## Ensemble Model Building 
 -  E_hv1
 -  E_hv2
 -  E_hv_flex
 
 ## Evaluating KNN, log_reg and GNB, E_hv1, E_hv2, E_hv_flex
 
 ## Further work 
 -soft voting classifer 
## How to use the tool developed 
