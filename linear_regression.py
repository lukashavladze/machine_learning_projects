import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading csv file
customers = pd.read_csv('Ecommerce Customers')
# checking data's head
print(customers.head())

# using seaborn to create jointplot to compare the Time on Website and  Yearly Amount Spent columns.
sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')
plt.show()

# using jointplot to create 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
plt.show()

# printing our data column names
print(customers.columns)

# making data for training and testing
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App',
               'Time on Website', 'Length of Membership']]

# splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# creating instance of Linearregression model named lm
lm = LinearRegression()

# training our model
lm.fit(X_train, y_train)

# printing coefficients of our model
print()
print(lm.coef_)

# making predictions
predictions = lm.predict(X_test)

# creating scatterplot of the real test values versus predicted values
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (real Values)')
plt.ylabel('Predicted Values')
plt.show()

# evaluating our model
