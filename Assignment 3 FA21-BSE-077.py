#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 25-11-23
# CSC461 – Assignment3 – Machine Learning
# Maryam Yousaf
# FA21-BSE-077
# Question 1 to 4 Machine Learning Algorithm with different validations


# In[2]:


# Q1: Provide responses to the following questions about the dataset.
# 1. How many instances does the dataset contain?
# A1. 110

# 2. How many input attributes does the dataset contain?
# A2. 7 height,weight,beard,hairlength,shoesize,scarf,eye-colour

# 3. How many possible values does the output attribute have?
# A3. 2 Male,Female

# 4. How many input attributes are categorical?
# A4. 4 beard,hairlength,scarf,eye-colour

# 5. What is the class ratio (male vs female) in the dataset
# A5. Male Ratio: No.of Male/ Total Instances 62/110 = 0.56
#     Female ratio: No.of Females/Total Instances 48/110 = 0.43


# In[3]:


# Q2: Apply Logistic Regression, Support Vector Machines, and Multilayer Perceptron classification  algorithms (using Python) on the gender prediction dataset with 2/3 train and 1/3 test split ratio and answer the following questions
# 1. How many instances are incorrectly classified?
get_ipython().system('pip install scikit-plot')


# In[4]:


from sklearn import preprocessing
import pandas as pd

from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#import ML evaluation metrics

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
# for question 3
from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeavePOut


#import scikitplot to plot confusion matrix

import scikitplot as skplt
data = pd.read_csv('gender-prediction.csv')
print(data)


# In[7]:


x = data.drop('gender', axis=1)  #input
y = data['gender']               #output
#print(x)
#print(y)   seperate input attributes from output
labels = preprocessing.LabelEncoder()
for column in data.columns:
    # Check if the column dtype is object categorical
    if data[column].dtype == 'object':
        # then convert it into numerical
        data[column] = labels.fit_transform(data[column])
#x_encoded = labels.fit_transform(x)
y_encoded = labels.fit_transform(y)


# In[8]:


X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model1 = LogisticRegression(max_iter=1000)
model2 = MLPClassifier(max_iter=1000, random_state=42)  #max iteration set to 1000 in order to avoid the error and setting random seed to make the initialization reproducible across runs.
model1.fit(X_train,Y_train)
prediction1 = model1.predict(x_test)
logic_acc = accuracy_score(y_test, prediction1)*100
print("Results for 2/3 train-test split:")
print("Logistic Regression Accuracy:", logic_acc)
model2.fit(X_train,Y_train)
prediction2 = model2.predict(x_test)
MLP_acc = accuracy_score(y_test, prediction2)*100
print("Results for 2/3 train-test split:")
print("MultiLayer Perception Accuracy:", MLP_acc)


# In[9]:


from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train,Y_train)
prediction3 = svm_model.predict(x_test)
SVM_acc = accuracy_score(y_test, prediction3)*100
print("Results for 2/3 train-test split:")
print("Support Vector Machine Accuracy:", SVM_acc)


# In[10]:


incorrectly_classified1 =  (100 - logic_acc)
incorrectly_classified2 = (100 - MLP_acc)
incorrectly_classified3 = (100 - SVM_acc)
print("Incorrectly Classified Instance using 67/33 split Logical regression:", int(incorrectly_classified1))
print("Incorrectly Classified Instance using 67/33 split Support Vector Machine:", int(incorrectly_classified3))
print("Incorrectly Classified Instance using 67/33 split MultiLayer Preception:", int(incorrectly_classified2))


# In[11]:


# 2. Rerun the experiment using train/test split ratio of 80/20. Do you see any change in the results? Explain
# A2. yes change is noticable, increasing the size of train set can help the model generalize to unseen data 
#     therefore, The number of instances incorrectly classified is affected by size of the testing set. A smaller testing set may lead to more variability in the evaluation metric.
X_train2, x_test2, Y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=42)
model_L = LogisticRegression(max_iter =1000)
model_L.fit(X_train2,Y_train2)
prediction_L = model_L.predict(x_test2)
logic_L_acc = accuracy_score(y_test2, prediction_L)*100
print("Results for 80/20 train-test split:")
print("Logistic Regression Accuracy:", logic_L_acc)


# In[12]:


model_M = MLPClassifier(max_iter=1000, random_state=42) 
model_M.fit(X_train2,Y_train2)
prediction_M = model_M.predict(x_test2)
MLP1_acc = accuracy_score(y_test2, prediction_M)*100
print("Results for 80/20 train-test split:")
print("MultiLayer Perception Accuracy:", MLP1_acc)


# In[13]:


svm_model1 = SVC()
svm_model1.fit(X_train2,Y_train2)
prediction_S_3 = svm_model1.predict(x_test2)
SVM1_acc = accuracy_score(y_test2, prediction_S_3)*100
print("Results for 80/20 train-test split:")
print("Support Vector Machine Accuracy:", SVM1_acc)
incorrectly_classified4 =  (100 - logic_L_acc)
incorrectly_classified5 = (100 - MLP1_acc)
incorrectly_classified6 = (100 - SVM1_acc)
print("Incorrectly Classified Instance using 80/20 split Logical regression:", int(incorrectly_classified4))
print("Incorrectly Classified Instance using 80/20 split Support Vector Machine:", int(incorrectly_classified5))
print("Incorrectly Classified Instance using 80/20 split MultiLayer Preception:", int(incorrectly_classified6))


# In[14]:


# 3. Name 2 attributes that you believe are the most “powerful” in the prediction task. Explain why?
# A3. Beard and Scarf are the most powerful in gender prediction task as Beards are more commonly associated with males and scarf with females
# The ability to detect and analyze the presence of beard and scarf can serve as a powerful feature for gender prediction.Models trained on factors, including the presence or absence of a beard or scarf, can learn patterns that distinguish between male and female individuals.


# In[15]:


# 4. Try to exclude these 2 attribute(s) from the dataset. Rerun the experiment (using 80/20 train/test split),did you find any change in the results? Explain
# A4. if the number decresed then indeed the attribute were essential for prediction but in this case model perform similarly because the dropped attributes were causing overfitting
X = data.drop(['beard', 'scarf','gender'], axis=1)
Y = data['gender']


# In[16]:


label = preprocessing.LabelEncoder()
for column in data.columns:
    # Check if the column dtype is object categorical
    if data[column].dtype == 'object':
        # then convert it into numerical
        data[column] = label.fit_transform(data[column])
#x_encoded = labels.fit_transform(x)
y_encoded = label.fit_transform(y)


# In[17]:


X_train3, x_test3, Y_train3, y_test3 = train_test_split(x, y, test_size=0.2, random_state=42)
model_L1 = LogisticRegression(max_iter =1000)
model_L1.fit(X_train3,Y_train3)
prediction_L1 = model_L1.predict(x_test3)
logic_L_acc1 = accuracy_score(y_test3, prediction_L1)*100
print("Results for 80/20 train-test split:")
print("Logistic Regression Accuracy:", logic_L_acc1)


# In[18]:


model_M1 = MLPClassifier(max_iter=1000, random_state=42) 
model_M1.fit(X_train3,Y_train3)
prediction_M1 = model_M1.predict(x_test3)
MLP1_acc1 = accuracy_score(y_test3, prediction_M1)*100
print("Results for 80/20 train-test split:")
print("MultiLayer Perception Accuracy:", MLP1_acc1)


# In[19]:


svm_model2 = SVC()
svm_model2.fit(X_train3,Y_train3)
prediction_S_4 = svm_model2.predict(x_test3)
SVM1_acc2 = accuracy_score(y_test3, prediction_S_4)*100
print("Results for 80/20 train-test split:")
print("Support Vector Machine Accuracy:", SVM1_acc2)
incorrectly_classified7 =  (100 - logic_L_acc1)
incorrectly_classified8 = (100 - MLP1_acc1)
incorrectly_classified9 = (100 - SVM1_acc2)
print("Incorrectly Classified Instance using 67/33 split Logical regression:", int(incorrectly_classified7))
print("Incorrectly Classified Instance using 67/33 split Support Vector Machine:", int(incorrectly_classified8))
print("Incorrectly Classified Instance using 67/33 split MultiLayer Preception:", int(incorrectly_classified9))


# In[20]:


# Question 4 using Gaussian Naïve Bayes classification algorithm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
new_data = pd.read_csv('gender-prediction-update.csv')
m = new_data.drop('gender', axis=1)
n = new_data['gender']
label_encoder = preprocessing.LabelEncoder()
for k in m.columns:
    if m[k].dtype == 'object':
        m[k] = label_encoder.fit_transform(m[k])
    
n_encoded = label_encoder.fit_transform(n)

M_train, m_test, N_train, n_test = train_test_split(m, n, test_size=10, random_state=42)
classifier = GaussianNB()
classifier.fit(M_train, N_train) # Train model using all instances
n_pred = classifier.predict(m_test) # prediction on 10 instances

accuracy = accuracy_score(n_test, n_pred)
precision = precision_score(n_test, n_pred, average='weighted')
recall = recall_score(n_test, n_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[21]:


# Question 3 Apply Random Forest classification algorithm (using Python) on the gender prediction dataset with Monte 
# Carlo cross-validation and Leave P-Out cross-validation. Report F1 scores for both cross-validation strategies
random_model = RandomForestClassifier()
split = ShuffleSplit(test_size = 0.33,train_size = 0.67,n_splits = 5,random_state = 42)   # use standard 2/3 and 1/3 split
f1_scores = cross_val_score(random_model, x, y, cv=split, scoring='f1')
for iteration, f1 in enumerate(f1_scores, start=1):
    print(f"Iteration {iteration}: F1 Score = {f1}")
average_f1 = sum(f1_scores) / len(f1_scores)      # take an average of f1 score 
print(f"\nAverage F1 Score Across Iterations: {average_f1}")   
# Result indicate the model has a high balance between precision and recall


# In[22]:


p_iteration = 5
leave_p_out = LeavePOut(p_iteration)
f1_scores1 = []


# In[ ]:


for i,j in leave_p_out.split(x):
    X_train, x_test = x.iloc[i], x.iloc[j]
    Y_train, y_test = y.iloc[i], y.iloc[j]

    random_model.fit(X_train, Y_train)
    y_pred = random_model.predict(x_test)
    f1 = f1_score(y_test, y_pred,zero_division = 1)
    f1_scores1.append(f1)


# In[ ]:


for item, f1 in enumerate(f1_scores1, start=1):
    print(f"Iteration {item}: F1 Score = {f1}")

Paverage_f1 = sum(f1_scores1) / len(f1_scores1) if len(f1_scores1) > 0 else 0
print(f"\nLeave P-out Average F1 Score Across Iterations: {Paverage_f1}")


# In[ ]:




