import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading csv file
customers = pd.read_csv('Ecommerce Customers')
# checking data's head
print(customers.head())

# using seaborn to create jointplot to compare the Time on Website and  Yearly Amount Spent columns.
sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')