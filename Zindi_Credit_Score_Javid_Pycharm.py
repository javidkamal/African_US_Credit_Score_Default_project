import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Load data
df_train = pd.read_csv("Train.csv")

# Extracting the target variable
y = df_train['target']
y=y.values

# Getting rid of unused columns
df_train.drop(columns = ['target', 'ID','customer_id','country_id','tbl_loan_id','lender_id', 'disbursement_date','due_date'], axis=1, inplace=True)

# making dummy variables of train and test data dataframes
dummies_loan_type = pd.get_dummies(df_train['loan_type'])
dummies_repeat_type = pd.get_dummies(df_train['New_versus_Repeat'])

# Concatenating dataframe with dummy variables
df = pd.concat([df_train, dummies_repeat_type, dummies_loan_type], axis=1)

# Getting rid of unused columns
df.drop(columns = ['loan_type', 'New_versus_Repeat'], axis = 1, inplace = True)

# Equating dataframes, extracting the columns
X=df
X=X.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Shapes
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape, "X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

# Random Forest Classifier
# Fitting the Random Forest classifier model
model_random_forest_classifier = RandomForestClassifier(random_state=1)
model_random_forest_classifier.fit(X_train, y_train)

# Predicting the Random Forest classifier model
y_hat_rfc = model_random_forest_classifier.predict(X_test)

# Evaluation metrics of Random Forest classifier
accuracy_random_forest_classifier = accuracy_score(y_test, y_hat_rfc)
print("Accuracy score of Random Forest Classifier: ", accuracy_random_forest_classifier)
f1_random_forest_classifier = f1_score(y_test, y_hat_rfc)
print("F-1 score of Random Forest Classifier: ", f1_random_forest_classifier)

# Logistic regression
# Fitting the Logistic Regression model
model_logistic_regression = LogisticRegression(random_state=0, max_iter=10000)
model_logistic_regression.fit(X_train, y_train)

# Predicting the Logistic Regression model
y_hat_lr = model_logistic_regression.predict(X_test)

# Evaluation metrics of Logistic Regression
accuracy_logistic_regression = accuracy_score(y_test, y_hat_lr)
print("Accuracy score of Logistic Regression: ", accuracy_logistic_regression)
f1_logistic_regression = f1_score(y_test, y_hat_lr)
print("F-1 score of Logistic Regression: ", f1_logistic_regression)

# Confusion matrix
cm = confusion_matrix(y_test, y_hat_lr)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification report: ")
print(classification_report(y_test, y_hat_lr))

# Get coefficients and intercept of Logistic Regression
coefficients = model_logistic_regression.coef_  # Coefficients for the features
intercept = model_logistic_regression.intercept_  # Intercept term
print("Logistic regression Coef: ", coefficients)
print("Logistic regression Intercept: ", intercept)
print("P(y=1∣X)= 1/(1+ ((e) ^ (-)(intercept + coefficient_1 * x_1 + coefficient_2 * x_2")
