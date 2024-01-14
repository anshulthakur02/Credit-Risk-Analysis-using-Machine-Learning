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

Before  getting into model training, I need to make sure that all of our variables are numerical given that some of the models I am going to use cannot operate on label data. I will simply do this using the one-hot encoding method.




