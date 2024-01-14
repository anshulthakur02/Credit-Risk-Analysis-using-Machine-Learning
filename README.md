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
Now that we know what kind of variables we’re dealing with here, let’s get to the nitty-gritty of things.

### Data exploration and preprocessing
First, I checked for missing values in the dataset.
