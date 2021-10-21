# Identifying-Player-Position
 

## Section 1: Project overview 
 - Created a tool that enables an NBA scout to identify an NBA player's position based on their per 36minutes statistics using 6 classification models. 
 - Data for this project was taken from basketballreference.com using the scraper developed from an earlier [Data Pipeline](https://github.com/favourumeh/DATA-PIPELINE) project 
 - This project uses three different classification algorithms: Naive Bayes, Logistic Regression and K Nearest Neighbours
 - In total six models were created:
    - three models used the individual algorithms mentioned above. They are called: 'KNN', 'log_reg' and 'GNB'
    - three more models were created using an ensemble approach. These ensemble models use hard voting with some minor tweaks which are explained in section 9. They are called: 'E_hv1', 'E_hv2' and 'E_hv_flex' 
 -  All models were evaluated by comparing each other against three classification metrics : precision, recall and accuracy. Their confusion matrices were also analysed to gauge how well each model performed for different labels (/classes). 
 -  This is a summary of the performance for each model:                                       
    ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main//Evaluation%20images/Accuracy%2C%20Precision%2C%20Recall%20for%20all%20models%20for%20evaluation%20data.png)                                 
 -  Feel free to try out the tool created by cloning the repository and opening the file: 'Testing_All_Models.py'. Follow the flowchart in the **Section 12** of this README for further guidance.  

 
 ## Section 2: Python version and packages 
Python Version: **3.8.3**

Packages: pandas, numpy, sklearn, matplotlib, seaborn, pickle


 ## Section 3: Project context

 Before diving into the project details, it is important to understand the independent variables (features) and the dependent variable (classes/ labels). 
 
 ### Dependent variable 
 - The dependent variable is player position (or 'Pos')
 - It is made up of 5 classes: 
    - 'C' = Centre
    - 'PF' = Power Forward
    - 'PG' = Point Guard
    - 'SF' = Small Forward
    - 'SG' = Shooting Guard
 
 ### Independent Variables 
 - The independent variables are per 36 minutes player statistics. 
 - The ones considered for this project are
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/images%20dump/feature%20definitions.png)
 - Go to the file **'Features_Explained.docx'** for further information behind these features. 

 
 
 ## Section 4: Data Splitting 
 The data used for this report was split into three sets: **1) Training set(70%); 2) Testing set(20%) and 3) Evaluation set (10%)**. 
 
 The non-ensemble models were trained and optimised using hyper parameter tuning and/or feature selection for the training set. As a result of this these models could be evaluated based on the testing and evaluations sets. 
 
  Making improvements to the ensemble model required the use of data different to the training set because KNN involves telling the model the correct prediction and the non-ensemble models used were already optimised for the training set. The testing set was, thus, used to 'train' the ensemble model. 
  
  All models were evaluated using the evaluation set as this data was totally unseen by both sets of models. 


 
  ## Section 5: Feature Scaling
  - Data inputted into KNN and Log_reg models were scaled using a min-max scaler:
     - KNN modelling required feature scaling because it involved the calculation of (Euclidean) distances between data points 
     - Log_reg required scaling to speed up convergence of solvers used to find maximum likelihood (e.g. 'Newton-cg')
    
  - Data inputted into the GNB model did not require feature scaling as this model works by determining the approximate gaussian distributions for each label for a given feature. 

  - The training, testing and evaluation datasets were all scaled separately using the same scaling procedure. This was done to avoid any passage of information between the different datasets. 


  
 ## Section 6: Feature Selection 
 
  - At the beginning of this stage the features considered were as many as 22. Features: **'FG', 'FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per',
       'eFG_per', 'FT', 'FTA', 'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS'**
  
  ### Feature selection for Gaussian Naive Bayes and K-Nearest Neighbours 
  
   - As both models are sensitive to 'noisy' (irrelevant) features their tuning process required the detection and removal of any irrelevant features. 
   - The goal of the feature selection was to narrow down the 22 features to the ones that maximised the classification accuracy for a 5-fold cross validation of the training set. Accuracy was used as the judgement metric because the labels (/classes) -- whilst not equal-- were fairly balanced. 

       - This was done in 2 steps:
      
         **1) Graphical Assessment: Equivalent Gaussian distribution** 
         
           - The mean and standard deviation of a feature for a given label was calculated. 
           - This was then used to plot an equivalent gaussian distribution to represent each label's feature distribution. Two examples of this type plot are shown below for the BLK and FT features : 
             ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/Spread%20of%20Labels%20for%20BLK%20and%20FT%20features.png)
             
           - From the images above, it is evident that the distribution of the BLK feature for each label is more spread out than for the FT feature. This indicates that BLK is better at predicting a player's position than FT. Additionally, it suggests that the GNB and KNN model will perform better without the FT feature because each model's prediction will become less confident due to the inability of FT to differentiate the labels. 
           
           - Graphical assessments are a good starting point in deciding which features were obviously redundant. The next step will quantify how much better the models are without each feature using Relative Standard Deviation (RSD). 
      
             **Note: The feature distribution for labels were not necessarily normal. The image above is a schematic to highlight the spread between the labels for a given feature.** 
      
         **2) Quantitative Assessment: Relative Standard Deviation(RSD)**
         
           - RSD was used as a measure of the relative spread of labels in a given feature. It is calculated by 1) Calculating the mean of a given feature for each label; 2) Calculating the average spread of the labels relative the mean of all label means for a given feature. 
           - Note: The 'average spread relative to the mean of all label means' was used  to capture the spread as opposed as a simple mean spread calculation because a set of large numbers will naturally have higher spreads despite not actually being more spread out than a set of small numbers. 
           - Once the RSD was calculated for all features, the features were ordered based on it. A slice of the features with top 3 highest RSD and lowest RSD are shown below
           ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Naive%20Bayes/final%20images/Relative%20standard%20deviation%20rankings.png)
           - To narrow down the most important features a 5-fold cross validation was conducted using the training set. All 22 features were inputted into a model(i.e. KNN or GNB) and then the mean cross validation accuracy was calculated. 
           
           - The process was repeated again and again. With each new iteration the feature with the worst RSD (e.g. FT for the 1st iteration) was removed and the model's accuracy was recalculated. This was done until the model was left with one feature
           - In the end the combination of features that produced the highest mean cross validation accuracy was deemed the most relevant. An example of the top 4 variable combinations with the highest accuracy and bottom 6 combination for KNN with number or neighbours = 5 is shown below 
           ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/KNN%20variable%20combo%20accuracy%20feature%20selection.png)
                 
 
  ### Feature Selection for Logistic Regression
  
 As Log_reg is a regression model it had to follow certain assumptions such as: 1) No multicollinearity; 2) Exogeneity. Correlation plots and Variance Inflation Factor calculations were used to assess the multicollinearity. Variables with correlation coefficients >0.7 and VIF> 5 were removed. 
 
 **Correlation Plot**
  ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Logistic%20Regression/final%20images/correlation_feature_selection.png)

  ### Feeatures used by each Model
  After the feature selection process these are the features used by each model:
  
  **KNN**: '3P', '3PA', '3P_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PF'
  
  **log_reg**: 'FG_per', '3PA', '3P_per', '2PA', '2P_per', 'FTA', 'FT_per', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
  
  **GNB**: 'FGA', 'FG_per', '3P', '3PA', '3P_per', '2P', '2PA', '2P_per', 'FTA', 'FT_per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
  
 ## Section 7: Tuning and Hyperparameter Tuning
 
   ### Gaussian Naive Bayes 
   
 Due to the simplicity of this model the only tuning conducted was the removal of 'noisy' data which has already been explained in the Feature Selection section.  

   ### K-Nearest Neighbours
   
 As well as the removal of 'noisy' data, this model was tuned by adjusting the number of neighbours (k-value) and changing the weight function used to in the prediction from 'uniform' to 'distance'. 
 
 A uniform weight function simply chooses the modal label(/class) from the k neighbours. The distance weight function chosen for this model places more emphasis (weight) on labels that are closer to the queried point by setting the weight as the inverse (Euclidean) distance between the query point and the neighbours. 

To gauge the effect of adjusting the K-value and weight function a 5-fold cross validation was conducted on the training data and the mean accuracy of the 5 models was calculated. The figure below shows the effect of the weight function used and k-value on the mean  accuracy. 

![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/tuning_k.png)

From the figure above it is evident that a K-value = 24 and the distance weight function was the most optimal combination as it produced the highest mean cross validation accuracy. 

   ### Logistic Regression
   
 SKLearn's Logistic Regression model has built-in ridge and lasso regularisation. It also offers different solvers that determine the maximum likelihood. To save on time GridsearchCV was used to try different combinations of solvers, types of regularisation(ridge and lasso) and inverse regularisation strengths. Twenty inverse regularisation strength values were uniformly distributed in log space from 10^-4 to 10^4. The mean accuracy and mean log loss of a 5-fold cross-validation of the training set was used to gauge the optimal combination. 
 
The optimal combination used the 'sag' solver, with ridge regularisation and an inverse regularisation strength of 206.913. The mean cross validation accuracy for this combination was 0.6991 and the log loss was 0.7512. 

 
 ## Section 8: Evaluating KNN, log_reg and GNB
 As the **test data** was not used in the training or tuning of these models. It was used to compare the performance against Accuracy, Precision and Recall 
 
 ### Broad Analysis on Accuracy, Precision and Recall 
  - The table below gives the mean accuracy and weighted Precision and Recall for each model  
![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Accuracy%2C%20Weighted%20precison%20and%20recall%20for%20non-ensemble%20models.png)

 - This table below gives the Precision and Recall for each label and for each model
![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Accuracy%20and%20Precision%20KNN%2C%20Log_reg%2C%20GNB.png)
   
- Brief Comments:
  - From the tables above, it is evident that the log_reg model performs the best overall in terms of precision, accuracy and recall. It ranks in the top 2 for precision and recall for all labels and has the highest overall accuracy, weighted precision and weighted recall.
  - All Models are good predictors of C and PG with precisions of at least 0.64. The recall scores for these labels were even better (at least 0.80). This is to be expected as the role played by these types of players are well-defined in the data chosen for this project 
  - All models struggled with predicting SF and PF labels as no model reached over 0.6 precision
  - To understand why some labels, have better precision and recall than others it is worth analysing the confusion matrices for each model.
 
 ### Confusion Matrix Analysis
 The figure below is the confusion matrices for all models: 
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Confusion%20matrices%20for%20all%20non-ensemble%20models%20for%20testing%20data.png)
 
 Brief Comments:
 - The reason why KNN and Log_reg predict the PF label so poorly is because it is confused for the SF label. 
 - GNB isn't as poor a predictor of the PF label as KNN and Log_reg, but PF is noticeably underpredicted. This underprediction is caused by overprediction of the C and SF labels which has caused dismal PF recall (0.36). More PFs are classed as Cs by the GNB model than as PFs. 
 - SF recall was dismal across all models. Both KNN and log_reg label more SFs as PFs than SFs. Despite having the highest SF recall out of all models more SFs are labelled as SGs and PFs than SFs for the GNB model. 
 
 ### Contextual observation and Final comments:
 Whilst not having amazing classification accuracy the models may have picked up on the fluidity of certain positions in basketball particularly SF and PF. It is this fluidity in positional play that caused lower recall scores for these positions. Additionally, a player's positional classification is not based on any objective metric rather a consensus amongst the player and their observers (coaches, scouts, audience etc). This consensus is typically based on who the player's positional play most resembles historically. However, not every player will fit into a particular box. It is possible for a player's positional play to be different to their identification and the models may capture this.
 
 The following section will outline the creation of ensemble models which will establish different hard voting systems which will enable the models to come to an agreement on a player's position.  
 
 
 
 ## Section 9: Ensemble Model Building 
 As there was an odd number of algorithms used for this project ensemble models (E_hv1, E_hv2 and  E_hv_flex) were produced using the non-ensemble models (KNN, log_reg and GNB) optimised for the training set. This was done with the belief that ensemble models would help generalise class prediction by inheriting the best traits of the individual models. 
 
 All the ensemble models use hard voting classification and were fed the test data. For this situation there were 3 possible voting scenarios:
 
  - Scenario 1: Consensus Vote (54.8% of test set):
    -  1A: All three non-ensemble models agree on the label and the label is correct (42.46% of test data)
    -  1B: All three non-ensemble models agree on the label and the label is incorrect (12.34% ...)
 
  - Scenario 2: Majority Vote(41.39% ...):
    -  2A: Two non-ensemble models agree on a label and the majority decision is correct (21.09% ...)
    -  2B: Two non-ensemble models agree on a label and the majority decision is incorrect (20.3% ...)

  - Scenario 3:'Hung Parliament'--The non-ensemble models do not agree on a label (3.82% ...):
    -  3A: 'Hung Parliament' and one of the non-ensemble models has chosen the correct label (3.55% ...)
    -  3B: 'Hung Parliament' and none of the non-ensemble models has chosen the correct label (0.27% ...)

Scenario 1B and 3B were lost-cause scenarios as they are on the extreme ends of the voting spectrum. At one end the ensemble model is wrong, but highly confident in its classification whilst on the other end the model is wrong and completely unsure of its classification. For both scenarios there is no good resolution, however, it could be argued that if all models reach a consensus then the player's positional play may be different to their current identification. In this case, the ensemble model's classification is sound, and this player is miscategorised. However, there is no way of truly knowing if this is the case without a significant influx of relevant features and a retraining of the non-ensemble models. Scenario 3B is concerning, but very rare (only 0.27% of the test data fell under this scenario).  

The E_hv1 model was used as a baseline ensemble model whilst E_hv2 and E_hv_flex models were 'trained' using the testing set to maximise accuracy. This was achieved by using information from their performance in the test set to guide their voting biases (see the subheadings below for more information). Due to this training the E_hv2 and E_hv_flex models can only be evaluated using the evaluation set as this dataset is unseen by either model. 
 
 ### E_hv1
 This model accepts the consensus and majority votes from scenario 1 and 2. For scenario 3 the model randomly chooses a non-ensemble model to decide. 
 
 **Advantage:** This voting system is fair 
 
 **Disadvantage:** The results are not repeatable or reproducible due to the random selection process for Scenario 3. 
 
 ### E_hv2
 This model accepts the consensus and majority votes from scenario 1 and 2. For scenario 3 the non-ensemble model with the highest accuracy in this scenario for the testing set was chosen. The most accurate model for scenario 3 was log_reg at 45%. 
  
 **Advantage:** This voting system will yield a higher accuracy than E_hv1
 
 **Disadvantage:** This model is biased towards the optimised log_reg model for scenario 3. 
 
 ### E_hv_flex
 This model deviates slightly from hard voting principles. It accepts the consensus vote from scenario 1 and adopts the same voting for scenario 3 as E_hv2. The majority vote for scenario 2 is upheld except for the situation where KNN and log_reg predict a PF label whilst GNB predicts a SF label.  In this situation GNB’s SF prediction is upheld because it  was found that this prediction had a precision of 49.8% whilst the PF prediction by KNN and log_reg had a precision of 34.2%. In doing this the model inherits GNB's superior recall for the SF label, but this comes with a trade-off as the model's PF recall reduces (see confusion matrices in the next section to see this trade-off). Nevertheless, the increased recall of the SF label should offset the reduced recall of the PF label. 

**Advantage:** This voting system will yield a higher accuracy than E_hv2
 
**Disadvantage:** This model is biased towards the optimised log_reg model for scenario 3. It is also biased towards the optimised GNB model for scenario 2 (where KNN and log_reg predict the PF label whilst GNB predicts the SF). 
 
 
 
 ## Section 10: Evaluating KNN, log_reg and GNB, E_hv1, E_hv2, E_hv_flex
 To evaluate all 6 models produced, the evaluation set was used as this data is unseen by all the models.  
 
 ### Accuracy, Precision and Recall 
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Accuracy%2C%20Precision%2C%20Recall%20for%20all%20models%20for%20evaluation%20data.png)
    
      
 ### Confusion Matrices
 
 #### 1) Non-Ensemble Models
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Confusion%20matrices%20for%20all%20non-ensemble%20models%20for%20evaluation%20data.png)

 #### 2) Ensemble Models
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Evaluation%20images/Confusion%20matrices%20for%20all%20ensemble%20models%20for%20evaluation%20data.png)

### Evaluation 

All non-ensemble models have low variance because the precision and recall scores are similar to those from the test dataset. Based on these findings, it can be assumed that the ensemble models will also have low variance.  

As expected, E_hv_flex appears to have the best overall precision, recall and accuracy. However, it is still just 2% better off than the best overall non-ensemble model, log_reg. Additionally, the confusion matrix for E_hv_flex and E_hv2 shows that E_hv_flex has improved SF recall at the cost of PF recall. 

There appears to be a limit to the precision, recall and accuracy that can be achieved by the current models. This limit may be caused by the features used in each of the optimised non-ensemble models. The current features could make it difficult for the non-ensemble models to differentiate the PF, SF and C classes. The purpose behind creating the ensemble models was the belief that they would inherit the good traits of the non-ensemble model, and this is evident in high precision and recall of the C and PG classes; but also evident are the poor distinction of the PF, SF and c classes. 

To improve the precision and recall of the models, stronger feature engineering is required. The addition of new features that show clear separation of the classes should improve the precision and recall for all classes much more than further tuning or ensemble model tweaks. 

 ## Section 11: Further work 
The below sub-sections are suggestions for improvements that could be made to improve precision and recall for all classes.

### Improving precision and recall 
 
This can be done by adding features which help to better distinguish the classes such as 'player height' or 'AST/TOV' ('Assist/Turnover'). Another possible feature is a score that takes into account the fact that the 'C' class was disproportionately affected by nulls in the '3P_per' feature. Roughly 33% of the C label had a null value for this feature compared to 15% for PF, 2.3% for SF, 0.7% for PG and 0.6% for SG. 
 
 ### Improve ensemble approach 
 A soft voting classifier may also prove better than a hard voting classifier for situations where the non-ensemble models have high uncertainty in a classification. However, for this to be fruitful the feature engineering suggestions need to be implemented.   
 
## Section 12: How to use the tool developed 
The tool developed allows the user to use all models created in this project (ensemble and non-ensemble) to predict the position of one or more players. 

The variable 'y_pred' in the code should give the prediction(s) made. When Identifying a single player, the prediction will also be printed in the console. When identifying more than one player the predictions will not be in the console instead there will be a classification report and confusion matrix. 

Follow the flow chart below to use the tool which is in the file 'Testing_All_Models.py' in the folder [Final Model_ Hard voting classifier](https://github.com/favourumeh/Identifying-Player-Position/tree/main/Ensemble/Final%20Model_%20Hard%20voting%20classifier). 

![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Ensemble/Final%20Model_%20Hard%20voting%20classifier/How%20to%20use%20the%20tool.png)
