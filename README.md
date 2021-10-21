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
  
  ### Feature selection for Gaussian Naive Bayes and K-Nearest Neighbours 
  
   - As both of these models are sensitive to 'noisey' (irrelevant) features their tunning process required the detection and removal of any irrelevant features. 
   - The goal of the features selection was to narrow down the 22 features to the ones that maximised the mean classifcation accuracy for a 5-fold cross validation of the training set. Accuracy was used as the judgement metric because the labels (/classes) -- whilst not equal-- were fairly balanced. 

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
                 
 
  ### Feature Selection for Logistic Regression
  
 As Log_reg is a regression model it had to follow certain assumptions such as: 1) No multicollinearity; 2) Exogeneity. Correlation plots and Variance Inflation Factor calculations were used to assess the multicolinearity. Variables with correlation coefficients >0.7 and VIF> 5 were removed. Exogeneity was upheld 
 
 **Correlation Plot**
  ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Logistic%20Regression/final%20images/correlation_feature_selection.png)
    
 ## Tuning and Hyperparameter Tuning
 
   ### Gaussian Naive Bayes 
   
 Due to the simplicity of this model the only tuning conducted was the removal of 'noisy' data which has already been explained in the Feature Selection section.  

   ### K-Nearest Neighbours
   
 As well as the removal of 'noisy' data, this model was tuned by adjusting the number of neighbours (k-value) and changing the weight function used to in the prediction from 'uniform' to 'distance'. 
 
 A uniform weight function simply choses the modal label(/class) from the k neighbours. The distance wieght function chosen for this model places more emphases (weight) on labels that are closer to the queried point by setting the weight as the inverse (Euclidean) distance between the query point and the neighbours. 

To gauge the effect of adjusting the K-value and weight function a 5-fold cross validation was conducted on the training data and the mean (mean) accuracy of the 5 models was calculated. The figure below shows the effect of the weight function used and k-value on the mean (mean) accuracy. 

![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/KNN/final%20images/tuning_k.png)

From the figure above it is evident that a K-value = 24 and the distance weight function was the most optimal combination as it produced the highest mean cross validation accuracy. 

   ### Logistic Regression
   
 SKLearn's Logistic Regression model has built-in ridge and lasso regularisation. It also offers different solvers that determine the maximum likelhood. To save on time GridsearchCV was used to try different combinations of solvers, types of regularisation(ridge and lasso) and inverse regularisation strenghts. Twenty inverse regularisation strength values were uniformly distributed in logspace from 10^-4 to 10^4. The mean (mean)accuracy and mean log loss of a 5-fold cross-validation of the training set was used to gauge the optimal combination. 
 
The optimal combination used the 'sag' solver, with ridge regularisation and an inverse regularisation strength of 206.913. The mean cross validation accuracy for this combination was 0.6991 and the log loss was 0.7512. 
 
 ## Evaluating KNN, log_reg and GNB
 As the **test data** was not used in the training or tuning of these models. It was used to compare the performance against Accuracy, Precision and Recall 
 
 ### Broad Analysis on Accuracy, Precision and Recall 
  - The table below give the mean accuracy and weighted Precision and Recall for each model  
![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Accuracy%2C%20Weighted%20precison%20and%20recall%20for%20non-ensemble%20models.png)

 - This table below gives the Precision and Recall for each label in each and for each model
![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Accuracy%20and%20Precision%20KNN%2C%20Log_reg%2C%20GNB.png)
   
- Brief Comments:
  - From the tables above its is evident that the log_reg model performs the best overall in terms of precision, accuracy and recall. It ranks in the top 2 for precision and recall for all labels and has the highest overall accuracy, weighted precsion and weighted recall.
  - All Models are good predictors of C and PG with precisions of at least 0.64. The recall scores for these labels were even better (at least 0.80). This is to be expected as the role played by these types of players are well-defined in the data chosen for this project 
  - All models struggled with predicting SF and PF labels as no model reached over 0.6 precision
  - To understand why some labels have better precision and recall than others it is worth analysing the confusion matrices for each model.
 
 ### Confusion Matrix Analysis
 The figure below is the confusion matrices for all models: 
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Confusion%20matrices%20for%20all%20non-ensemble%20models%20for%20testing%20data.png)
 
 Brief Comments:
 - The reason why KNN and Log_reg predict the PF label so poorly is because it is confused for the SF label. 
 - GNB isn't as poor a predictor of the PF label as KNN and Log_reg, but PF is noticeably underpredicted. This undeprediction is caused by overprediction of the C and SF labels which has caused dismal PF precision (0.36). More PFs are classed as Cs by the GNB model than as PFs. 
 - SF accuracy was dismal across all models. Both KNN and log_reg label more SFs as PFs than SFs. Despite having the highest SF recall out of all models more SFs are labelled as SGs and PFs than SFs. 
 
 ### Contextual observation and Final comments:
 Whilst not having amazing classification accuracy the models may have picked up on the fluidity of certain positions in basketball particularly SF and PF. It is this fluidity in positional play that caused lower recall scores for these positions. Addtionally, a player's positional classifcation is not based on any objective metric rather a concensus amongst the player and their observers (coaches, scouts, audience etc). This concensus is typically based on who the player's positional play most resembles historically. However, not every player will fit into particular box. It is possible for a player's positional play to be different to their identification and the models may capture this.
 
 The following section will outline the creation of ensemble models which will establish different voting systems which will enable the models to come to an agreement on a player's position.  
 
 
 
 ## Ensemble Model Building 
 As there was an odd number of algorithms used for this project ensemble models (E_hv1, E_hv2 and  E_hv_flex) were produced using the non-ensemble models (KNN, log_reg and GNB) optimised for the training set. This was done to create a centralised tool that inherits the best traits the individual models. 
 
 All the ensemble models use hard voting classification and were fed the test data. For this situation there were 3 possible voting senarios:
 
  - Scenario 1: Consensus Vote (54.8% of test set):
    -  1A: All three non-ensemble models agree on the label and the label is correct (42.46% of test data)
    -  1B: All three non-ensemble models agree on the label and the label is incorrect (12.34% ...)
 
  - Scenario 2: Majority Vote(41.39% ...):
    -  2A: Two non-ensemble models agree on a label and the majority decision is correct (21.09% ...)
    -  2B: Two non-ensemble models agree on a label and the majority decision is incorrect (20.3% ...)

  - Scenario 3:'Hung Parliament'--The non-ensemble models do not agree on a label (3.82% ...):
    -  3A: 'Hung Parliament' and one of the non-ensemble models has chosen the correct label (3.55% ...)
    -  3B: 'Hung Parliament' and none of the non-ensemble models has chosen the correct label (0.27% ...)

Scenraio 1B and 3B were lost-cause scenrios as they are on the extreme ends of voting spectrum. At one end the ensemble model is wrong, but highly confident in it classification whilst on the other end the model is wrong and completely unsure of its classifcation. For both scenrios there is no good resolution, however, it could be argued that if all models reach a concensus then the player's positional play may be different to their current identification. In this case ensemble model's classification is sound and this player is miscategorised. However, there is no way of truly knowing if this is the case. Scenraio 3B is concerning, but very rare (only 0.27% of the test data fell under this scenrio).  

The E_hv1 model was used as baseline ensemble model whilst E_hv2 and E_hv_flex models were 'trained' using the testing set to maximise accuracy. This was achived by using information from their perfomance in the test set as used to inform their voting biases (see the subheadings below for more information). Due to this training the E_hv2 and E_hv_flex models can only be evaluated using the evaluation set as this dataset is unseen by either model. 
 
 ### E_hv1
 This model accepts the concensus and majority votes from scenrio 1 and 2. For scenrario 3 the model randomly choses a non-ensemble model to decide. 
 
 **Advantage:** This voting system is fair 
 **Disadvantage:** The results are not repeatable or reproducible due to the random selection process for Scenrario 3. 
 
 ### E_hv2
 This model accepts the concensus and majority votes from scenrio 1 and 2. For scenrario 3 the non-ensemble model with the highest accuracy in this scenario for the testing set was chosen. The most accurate model for scenrio 3 was log_reg at 45%. 
  
 **Advantage:** This voting system will yield have a higher accuracy than E_hv1
 **Disadvantage:** This model is biased towards the optimised log_reg model for scenrio 3. 
 
 ### E_hv_flex
 This model deviates slightly from hard voting principles. It accepts the concensus vote from scenrio 1 and adopts the same voting for scenraio 3 as E_hv2. However, for scenrio 2 situations where KNN and log_reg assigned a PF label and GNB assigned a SF label. 
 
 It was found that in this scenario GNB's SF prediction had a precision of 49.8% whilst the PF prediction by KNN and log_reg had a precision of 34.2%. In doing this the model inherits GNB's superior recall for the SF label, but this comes with a trade-off as the model's PF recall reduces (see next section for visual aid of this trade-off). Nevertheless, this makes for a more balanced recall for all labels 

**Advantage:** This voting system will yield have a higher accuracy than E_hv2
 **Disadvantage:** This model is biased towards the optimised log_reg model for scenrio 3. It is also biased towards the optimised GNB model for scenario 2 (where KNN and log_reg predict the PF label whilst GNB Predicts the SF). 
 
 
 ## Evaluating KNN, log_reg and GNB, E_hv1, E_hv2, E_hv_flex
 To evaluate all 6 models produced the evaluation set was used as this data is unseen by all the models.  overall accuracy, precision and recall and the. 
 
 ### Analysis on Accuracy, Precision and Recall 
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Accuracy%2C%20Precision%2C%20Recall%20for%20all%20models%20for%20evaluation%20data.png)
 
   - As expected E_hv_flex appears to have to best overall precision, recall and accuracy. However, it is still just 2% better off the best overall non-ensemble model, log_reg. 
   - All non-ensemble models have low variance because the precision and recall 
   - 
     
      
      
      
      
      
      
      
      
      
 ### Analysis on confusion matrices
 
 
 #### 1) Non-Ensemble Models
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Confusion%20matrices%20for%20all%20non-ensemble%20models%20for%20evaluation%20data.png)

 #### 2) Ensemble Models
 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/Confusion%20matrices%20for%20all%20ensemble%20models%20for%20evaluation%20data.png)


 ## Further work 
 -improving accuracy
 
 Undergoing further model tunning or utilising different algorithms (e.g. Random Forest) may offer some further improvement the classifcation accuracy, but for noticeable  improvement stronger feature engineering is requrired. This can be done by  adding features which help to better distinguish the classes such a 'player height' or 'AST/TOV' ('Assist/Turnover'). Another possible feature is a 'three point score' as when processing the null values it was found that the 'C' class was disproportionately affected by nulls in the '3P_per' feature. Roughly 33% of the C label had a null value for this feature compared to 15% for PF, 2.3% for SF, 0.7% for PG and 0.6% for SG. From the confusion matrices it is evident that that all models had trouble differentiating PFs from Cs and SFs thus the 'three point score' should imporve this.  
 
 -soft voting classifer 
## How to use the tool developed 
