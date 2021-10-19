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
 
 The non-ensemble models were trained and optimised (i.e. using hyper parameter tuning or feature selection) using only the training set. As a result of this these models could be evaluated based on the testing and evaluations sets. 
 
  Making improvements to the ensemble model required the use of data different to the training set because KNN involves telling the model the correct prediction and the non-ensemble models used were already optimised for the training set. The testing set was, thus, used to 'train' the ensemble model (see ensemble section for more information). 
  
  All models were evaluated using the evaluation set as this data was totaly unseen by both sets of models. 

 
  ## Feature Scalling
  - Data inputed into KNN and Log_reg models were scaled using a min-max scaler:
     - KNN modelling required feature scalling because it involved the calculation of (Euclidean) distances between data points 
     - Log_reg required scalling to speed up convergence of solvers used to find maximum likelihood (e.g. 'Neton-cg')
    
  - Data inputted into the GNB model did not require feature scalling as this model works by determining the approximate gaussian distributions for each label for a given feature. 

  - The training, testing and evaluation datasets were all scaled seperately using the same scalling procedure. This was done to avoid any passage of information between the different datasets. 
   

  
 ## Feature Selection 
 
  - At the begining of this stage the features considered were as many as 22. Features: **'FG', 'FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per',
       'eFG_per', 'FT', 'FTA', 'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS'**
  
  - Feature selection for Gaussian Naive Bayes and K-Nearest Neighbours 
      - As both of these models are sensitive to 'noisey' (irrelevant) features their tunning process required the detection and removal of any irrelevant features. 
      - The goal of the features selection was to narrow down the 22 features to the ones that maximised the mean classifcation accuracy for a 5-fold cross validation of the training set. 
  
      - This was done in 2 steps:
      
         **1) Graphical Assessment: Equivalent Gausssian distribution** 
         
           - The mean and standard devition of a feature for a given label(i.e class/dependent variable) was calculated. 
           - This was then used to plot an equivalent gaussian distribution to represent each label's feature distribution. Two examples of this type plot is shown below for the BLK and FT features : 
             ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/Spread%20of%20Labels%20for%20BLK%20and%20FT%20features.png)
             
           - From the image above it is evident that the distribution of the BLK feature for each label is more speread out than for the FT feature. This indicates that BLK is a better at predicting a player's position than FT. Additionally, it suggests that the GNB and KNN model will perfrom better without the FT feature becasue each model's prediction will become less confident due to the inability of FT to differentiate the labels. 
           
           - Graphical assessments are a good starting point in deciding which features were obviously redundant. The next step will quantify how much better the models are without each feature using Relative Standard Deviation (RSD). 
      
             **Note: The feature distribution for labels were not necessarily normal. The image above is a schemeatic to highligh the spread between the labels for a given feature.** 
      
         **2) Quantitative Assessment: Relative Standard Deviation(RSD)**
         
           - RSD was used as a measure of the relative spread of labels in a given feature. It is calculated by: 1) Calculating the mean of a given feature for each label; 2) Calculating the average spread of the label relative the the mean of all label means for a given feature. 
           - Note: The 'average spread relative to the mean of all label means' was used  to capture the spread as opposed as a simple mean spread calculation becuase a larger set of numbers will naturally have higher spreads despite not actually being more spread out than a smaller set of numbers. 
           - Once the RSD was calculated for all features, the features were ordered based on it. A slice of the features with top 3 highest RSD and lowest RSD are shown below
           ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Naive%20Bayes/final%20images/Relative%20standard%20deviation%20rankings.png)
           - To narrow down the most important features a 5-fold cross validation was conducted using the training set. All 22 features were inputted into a model(i.e. KNN or GNB) and then The mean(mean) accuracy of the model was then calculated. The process was repeated again and again.
           
           - With each new iteration the feature with the worst RSD (e.g. FT for the 1st iteration) was removed and the model's accuracy was recalculated. This done until the model was left with one feature
           - In the end the combination of features that produced the highest (mean) cross validation accuracy was used deemed the most relevant. An example of the top 4 varibale combos with the highest accuracy and bottom 6 combos for KNN with number or neighbour = 5 is shown below 
           ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/KNN%20variable%20combo%20accuracy%20feature%20selection.png)
                 
 
  - Feature Selection for Logistic Regression
  
 As Log_reg is a regression model it had to follow certain assumptions such as: 1) No multicollinearity; 2) Exogeneity. Correlation plots and Variance Inflation Factor calculations were used to assess the multicolinearity. Variables with correlation coefficients >0.7 and VIF> 5 were removed. Exogeneity was upheld 
 
 **Correlation Plot**
  ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Logistic%20Regression/final%20images/correlation_feature_selection.png)
  
  **VIF**
  
  ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Logistic%20Regression/final%20images/VIF.png)
  
 ## Tuning and Hyperparameter Tuning
   ### Gaussian Naive Bayes 
   
 Due to the simplicity of this model the only tuning conducted was the removal of 'noisy' data which has already been explained in the Feature Selection section.  

   ### K-Nearest Neighbours
   
 As well as the removal of 'noisy' data, this model was tuned by adjusting the number of neighbours (k-value) and changing the weight function used to in the prediction from 'uniform' to 'distance'. 
 
 A uniform weight function simply choses the modal label(/class) from the k neighbours. The distance wieght function chosen for this model places more emphases (weight) on labels that are closer to the queried point by setting the weight as the inverse (Euclidean) distance between the query point and the neighbours. 

To gauge the effect of adjusting the K-value and weight function a 5-fold cross validation was conducted on the training data and the mean (mean) accuracy of all 5 models was calculated. The figure below shows the effect of the weight function used and k-value on the mean (mean) accuracy. 

![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/tuning_k.png)

   ### Logistic Regression
   
 
 ## Evaluating KNN, log_reg and GNB
 
 ## Ensemble Model Building 
 -  E_hv1
 -  E_hv2
 -  E_hv_flex
 
 ## Evaluating KNN, log_reg and GNB, E_hv1, E_hv2, E_hv_flex
 
 ## Further work 
 -soft voting classifer 
## How to use the tool developed 
