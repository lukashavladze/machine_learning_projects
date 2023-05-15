import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# getting data
df = pd.read_csv('KNN_Project_Data')

# standardize the Variables
scaler = StandardScaler()

# fit scaler to the features
scaler.fit(df.drop('TARGET CLASS', axis=1))

# transforming features into scaled version
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# converting the scaled features to a dataframe and checking if scaling worked
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

# training and testing
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'],
                                                    test_size=0.30)

# using KneighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# training our model
knn.fit(X_train, y_train)

# predictions and evaluations
pred = knn.predict(X_test)

# creating confussion matrics and classification report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# choosing a K value
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# creating plot using loop info
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# retraining with best K value
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
