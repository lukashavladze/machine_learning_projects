import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# getting data
loans = pd.read_csv('loan_data.csv')

# checking data
print(loans.info())

# exploratory Data analysis
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# the trend between FICO score and interest rate
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.show()

# creating lmplots to see if the trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
plt.show()

# making dummy variables to tranform data
# creating a fixed larger dataframe that has new feature columns with dummy variables
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# training decision tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# predictions and evaluations of decision tree
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# training random forest model
# creating instance of the randomforestclassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

predictions_r = rfc.predict(X_test)
print(classification_report(y_test, predictions_r))
print(confusion_matrix(y_test, predictions_r))



