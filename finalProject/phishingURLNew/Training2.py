import matplotlib as matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
from sklearn import metrics
import warnings
import pickle
warnings.filterwarnings('ignore')

#Loading data into dataframe

data = pd.read_csv("phishing.csv")
data.head()

#Shape of dataframe

data.shape


#Listing the features of the dataset

data.columns

#Information about the dataset

data.info()


# nunique value in columns

data.nunique()

#droping index column

data = data.drop(['Index'],axis = 1)

#description of dataset

data.describe().T

plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)
plt.show()


#pairplot for particular features

df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]
sns.pairplot(data = df,hue="class",corner=True);


data['class'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.title("Phishing Count")
plt.show()


X = data.drop(["class"],axis =1)
y = data["class"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#function to call for storing the results
def storeResults(model, a,b,c,d):
  ML_Model.append(model)
  accuracy.append(round(a, 3))
  f1_score.append(round(b, 3))
  recall.append(round(c, 3))
  precision.append(round(d, 3))

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'gamma': [0.1], 'kernel': ['rbf', 'linear']}

CNN = GridSearchCV(SVC(), param_grid)

# fitting the model for grid search
CNN.fit(X_train, y_train)

y_train_svc = CNN.predict(X_train)
y_test_svc = CNN.predict(X_test)


acc_train_svc = metrics.accuracy_score(y_train,y_train_svc)
acc_test_svc = metrics.accuracy_score(y_test,y_test_svc)
print("Support Vector Machine : Accuracy on training Data: {:.3f}".format(acc_train_svc))
print("Support Vector Machine : Accuracy on test Data: {:.3f}".format(acc_test_svc))
print()

f1_score_train_svc = metrics.f1_score(y_train,y_train_svc)
f1_score_test_svc = metrics.f1_score(y_test,y_test_svc)
print("Support Vector Machine : f1_score on training Data: {:.3f}".format(f1_score_train_svc))
print("Support Vector Machine : f1_score on test Data: {:.3f}".format(f1_score_test_svc))
print()

recall_score_train_svc = metrics.recall_score(y_train,y_train_svc)
recall_score_test_svc = metrics.recall_score(y_test,y_test_svc)
print("Support Vector Machine : Recall on training Data: {:.3f}".format(recall_score_train_svc))
print("Support Vector Machine : Recall on test Data: {:.3f}".format(recall_score_test_svc))
print()

precision_score_train_svc = metrics.precision_score(y_train,y_train_svc)
precision_score_test_svc = metrics.precision_score(y_test,y_test_svc)
print("CNN : precision on training Data: {:.3f}".format(precision_score_train_svc))
print("CNN : precision on test Data: {:.3f}".format(precision_score_test_svc))


#computing the classification report of the model

print(metrics.classification_report(y_test, y_test_svc))



storeResults('CNN model',acc_test_svc,f1_score_test_svc, recall_score_train_svc,precision_score_train_svc)

with open('CNN_model.pkl', 'wb') as file:
  pickle.dump(ML_Model, file)
print("Model saved as CNN_model.pkl and scaler saved as scaler.pkl")
# Save the scaler as well

