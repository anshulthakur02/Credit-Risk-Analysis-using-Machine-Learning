# Credit-Risk-Analysis-using-Machine-Learning

### Introduction
Credit default risk is simply known as the possibility of a loss for a lender due to a borrower’s failure to repay a loan.Machine learning algorithms have a lot to offer to the world of credit risk assessment due to their unparalleled predictive power and speed. In this project, I will be utilizing machine learning’s power to predict whether a borrower will default on a loan or not and to predict their probability of default. 

### Dataset
The dataset I'm using can be found on Kaggle and it contains data for 32,581 borrowers and 11 variables related to each borrower. Let’s have a look at what those variables are:

* Age — numerical variable; age in years
* Income — numerical variable; annual income in dollars
* Home status — categorical variable; “rent”, “mortgage” or “own”
* Employment length — numerical variable; employment length in years
* Loan intent — categorical variable; “education”, “medical”, “venture”, “home improvement”, “personal” or “debt consolidation”
* Loan amount — numerical variable; loan amount in dollars
* Loan grade — categorical variable; “A”, “B”, “C”, “D”, “E”, “F” or “G”
* Interest rate — numerical variable; interest rate in percentage
* Loan to income ratio — numerical variable; between 0 and 1
* Historical default — binary, categorical variable; “Y” or “N”
* Loan status — binary, numerical variable; 0 (no default) or 1 (default) → this is going to be our target variable
  
Now I know what kind of variables I am dealing with here, let’s get to the nitty-gritty of things.

### Data exploration and preprocessing
First, I will check for missing values in the dataset. The columns employment length and interest rate both have missing values. Given that the missing values represent a small percentage of the dataset, I will remove the rows that contain missing values.

Next, I will look for outliers in our dataset so I can remedy them accordingly. I will do this using the describe() method which is used for calculating descriptive statistics. Not only will it help identify outliers, but it will also give me a better understanding of how our data is distributed. I’ll also be using a scatterplot matrix, a grid of scatterplots used to visualize bivariate relationships between combinations of variables, to visually detect outliers. I will also look for outliers using a scatterplot matrix.

The variables 'Age', 'Employment Length' and 'income' have outliers that needs to be removed. So I am going to remove the outliers.

Given the nature of the dataset, I’d expect that we’re dealing with an imbalanced classification problem, meaning that we have considerably more non-default cases than default cases. I confirmed that this is indeed the case with 78.4% of our dataset containing non-default cases.

With this in mind, I’ll now further explore how loan status is related to other variables in the dataset.

Two things quickly stand out upon looking at this box plot. We can clearly see that those who don’t default have a lower loan to income ratio mean value across all loan grades; which doesn’t come as a surprise. Also that no borrowers with loan grade G were able to repay their loan.

Before  getting into model training, I need to make sure that all of our variables are numerical given that some of the models I am going to use cannot operate on label data. I will simply do this using the one-hot encoding method and then proceed to split the dataset into a train and test split.

### Model training and evaluation

In this section, I’ll be training and testing 3 models, namely KNN, logistic regression and XGBoost and then evaluate their performance at predicting loan defaults and their probability.

First, let's build the models and look at some evaluation metrics for assessing the model’s ability to predict class labels, i.e., default or no default.

We’ve identified earlier that we are dealing with an imbalanced dataset and so we need to make sure we’re using the appropriate evaluation metrics for our case. For this reason, we’ll be looking at the common Accuracy metric with a grain of salt. To illustrate why this is the case, accuracy calculates the ratio of total truly predicted values to the total number of input samples, meaning that our model would get pretty high accuracy by predicting the majority class but would fail to capture the minority class, default, which is no bueno. This is why the evaluation metrics that we’ll be focusing on to assess the classification performance of our models are Precision, Recall and F1 score.

Firstly, Precision gives us the ratio of true positives to the total positives predicted by a classifier where positives denote default cases in our context. Given that they’re the minority class in our dataset, we can see that our models do a good job at correctly predicting those minor instances. Moreover, Recall, a.k.a true positive rate, gives us the number of true positives divided by the total number of elements that actually belong to the positive class. In our case, Recall is a more important metric as opposed to Precision given that we’re more concerned with false negatives (our model predicting that someone is not gonna default but they do) than false positives (our model predicting that someone is gonna default but they don’t). Lastly, F1 Score provides a single score to measure both Precision and Recall. Now that we know what to look for, we can clearly see that XGboost performs the best across all 3 metrics. Although it scored better on Precision as opposed to Recall, it still has a pretty good F1 score of 0.81.

We’ll now have a look at ROC which is a probability curve with False Positive Rate (FPR) on the x-axis and True Positive Rate (TPR, recall) on the y-axis. The best model should maximize the TPR to 1 and minimize FPR to 0. With this said, we can compare classifiers using the area under the curve of the ROC curve, AUC, where the higher its value, the better the model is at predicting 0s as 0s and 1s as 1s.

We can once again see that XGBoost performs best as it has the highest AUC and so is the best classifier in distinguishing between default and no default classes.

So far, we’ve looked at each model’s ability to predict class labels, we’ll now evaluate their performance at predicting the probability of the sample belonging to the positive class, i.e., probability of default. For this task, we’ll use a Reliability Plot and Brier Score, where the former creates a diagram of the actual probabilities versus the predicted probabilities on a test set and the latter calculates the mean squared error between predicted probabilities and their respective positive class values. Given that the Brier Score is a cost function, a lower Brier Score indicates a more accurate prediction.

We can see from the above Brier Score that the XGBoost performs best once again, which doesn't come as a surprise by now, in comparison to other models. From this score and the plot, we conclude that our model is well-calibrated for probability prediction, meaning that predicted probabilities closely match the expected distribution of probabilities for each class, and so doesn't require further calibration.

It’s needless to say which model was chosen as our best performer at predicting class labels and the probability of default. 

### Feature Importance
Last but not least, it’s time to identify which features were most influential in predicting our target variable. For this task, we’ll be using feature importance by information gain which measures each feature’s contribution for each tree in XGBoost.

We can see from the figure above that rent as a home status, loan to income ratio and loan grade C are the top 3 most important features for predicting loan default and its probability.

 ### Conclusion 
To sum up, we’ve analyzed and pre-processed our data, trained and evaluated 3 models, namley KNN, logistic regression and XGBoost, for their ability to predict loan defaults and their probability. We used Precision, Recall, F1 and ROCAUC to evaluate the models’ performance at predicting class labels. We used these metrics in particular and discarded Accuracy given that we’re dealing with an imbalanced dataset. We also used a Reliability Plot and Brier Score to assess the calibration of our models. After having identified that XGBoost performed best on all metrics, we investigated which features were most important to our predictions using feature importance by information gain. With this said, we can round up our demonstration of how machine learning can be applied to the world of credit risk assessment.





