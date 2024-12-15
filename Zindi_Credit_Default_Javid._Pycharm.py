import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# To be used later for making submission.csv file
test_df = pd.read_csv('Test.csv')

# Getting rid of unwanted data columns not used in calculating the target value
train_data.drop(['ID', 'customer_id','country_id','tbl_loan_id','lender_id','New_versus_Repeat', 'disbursement_date', 'due_date'], axis=1, inplace=True)
test_data.drop(['ID', 'customer_id','country_id','tbl_loan_id', 'lender_id', 'New_versus_Repeat', 'disbursement_date', 'due_date'], axis=1, inplace = True)

# making dummy variables of train and test data dataframes
train_data_dummy = pd.get_dummies('loan_type')
test_data_dummy = pd.get_dummies('loan_type')

# Concatenating the dummy variable to the train data and test data dataframe
train_data = pd.concat([train_data, train_data_dummy], axis=1)
test_data = pd.concat([test_data, test_data_dummy], axis=1)

# Split features and extract target
y_train = train_data['target']

# making relevant dataframes to calculate the target variable
train_data.drop(['target', 'loan_type'], axis=1, inplace=True)
test_data.drop(['loan_type'], axis=1, inplace=True)

# Equating the dataframe
X_train =train_data

# Shape of X_train and y_train
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

# Random Forest classifier
# Train Random Forest classifier model
model_random_forest_classifier = RandomForestClassifier()
model_random_forest_classifier.fit(X_train, y_train)

# Make predictions of Random Forest classifier model
predictions_rfc = model_random_forest_classifier.predict(test_data)

# making use fo test_df to make dataframe with the predictions of Random forest classifier model
test_df['target'] = predictions_rfc
sub_rfc = test_df[['ID', 'target']]
sub_rfc.head()

# Converting dataframe to a csv file
sub_rfc.to_csv('submission_RFC.csv', index=False)

# Logistic Regression
# Train Logistic Regression model
model_logistic_regression = LogisticRegression(random_state=0, max_iter=10000)
model_logistic_regression.fit(X_train, y_train)

# Make predictions of Logistic Regression model
predictions_lr = model_logistic_regression.predict(test_data)

# Making use fo test_df to make dataframe with the predictions of Logistic Regression model
test_df['target'] = predictions_lr
sub_lr = test_df[['ID', 'target']]
sub_lr.head()

# Converting dataframe to a csv file
sub_lr.to_csv('submission_LR.csv', index=False)

# Get coefficients and intercept of Logistic Regression
coefficients = model_logistic_regression.coef_  # Coefficients for the features
intercept = model_logistic_regression.intercept_  # Intercept term
print("Logistic regression Coef: ", coefficients)
print("Logistic regression Intercept: ", intercept)
print("P(y=1∣X)= 1/(1+ ((e) ^ (-)(intercept + coefficient_1 * x_1 + coefficient_2 * x_2)")
