Folders and Files and their order to be executed:
/JAVID_CREDIT_SCORE_DEFAULT/Credit_Default_Javid.ipynb
/JAVID_CREDIT_SCORE_DEFAULT/Credit_Score_Javid.ipynb

Explanation of features and relevance:
1. Removing irrelevant features:
Removal of irrelevant features not required for calculation of the credit score or credit default which is a categorical variable, such as ID, Customer_id, Loan_id, country_id etc..
2. Features used for calculation of credit score default:
Relevant features used are duration of loan term, loan_type etc.. The loan_type was made into dummy variables such as Type_1, Type_2, Type_3 etc..
Finally, the credit default was evaluated manipulating the dataframes made from Train.csv and Test.csv files.

Environment and hardware:
1. For .ipynb files, Anaconda Jupyter lab, kernel: anaconda-2022.05-py39
https://anaconda.cloud/share/notebooks/aa97d10d-508d-4791-83d6-6a47bfeda22b/overview
https://anaconda.cloud/share/notebooks/b16fafb3-8a9c-406d-a7d8-700e87a1a81a/overview
2. For .py files, Pycharm IDE, python version: 3.9.7 on 16GB RAM PC.