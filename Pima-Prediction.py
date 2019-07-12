#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #for dataframe
import matplotlib.pyplot as plt #plot data
import numpy as np #for N-dimensional object support

#to plot inline instead of in seperate window 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"C:\Users\Anu Abey(user)\Documents\ML using Python\MachineLearningWithPython-master\Notebooks\data\pima-data.csv")


# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


df.tail(5)


# In[6]:


df.isnull().values.any()


# In[7]:


def plot_corr(df,size=11):
    """Function plots a graphical correlation matrix for each pair of columns in a dataframe
        Input:
            df: pandas dataframe
            size: vertical and horizontal size of the plot
        Displays:
            matrix of correlation between columns
            Blue(not well correlated)->dark red(very well correlated)
    """
    corr = df.corr() #dataframe correlation function
    fig,ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) #color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)),corr.columns) #draw x tick marks
    plt.yticks(range(len(corr.columns)),corr.columns) #draw y tick marks


# In[8]:


plot_corr(df)


# In[9]:


df.corr()


# In[10]:


del df['skin']


# In[11]:


df.head()


# In[12]:


plot_corr(df)


# In[13]:


diabetes_map = {True : 1, False : 0}


# In[14]:


df['diabetes']=df['diabetes'].map(diabetes_map)


# In[15]:


df.head(5)


# In[16]:


num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("No of true cases: "+str(num_true)+" "+str(num_true/(num_true+num_false)*100))
print("No of false cases: "+str(num_false)+" "+str(num_false/(num_true+num_false)*100))


# ## Splitting the data
# 70% data for training & 30% data for testing

# In[20]:


from sklearn.model_selection import train_test_split
feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_names = ['diabetes']
X = df[feature_col_names].values #predictor feaature columns
y = df[predicted_class_names].values #predicted class
split_test_size = 0.30 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)
#test size is 30%, 42 is the seed for the random number generator


# In[21]:


print (str(len(X_train)/len(df.index)*100))
print (str(len(X_test)/len(df.index)*100))


# ## Post-split data seperation

# Checking for hidden null values

# In[22]:


print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))


# In[24]:


from sklearn.impute import SimpleImputer
#Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0,strategy="mean",verbose=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


# ## Training Using Naive Bayes Algorithm

# In[26]:


from sklearn.naive_bayes import GaussianNB
#create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())


# ## Evaluate performance of training data

# In[33]:


# predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

#import the performance metrics library
from sklearn import metrics

#testing metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))


# ## Metrics

# In[32]:


print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))


# Confusion Matrix:
#     [[118 33]    [TrueNegative  FalsePositive]
#     [ 28 52] ]   [FalseNegative TruePositive]
# Columns ->Predicted values
# Left column => Predicted False
# Right column => Predicted True
# Rows -> Actual values
# Top row => Actual False
# Bottom row => Actual True

# Classification Report:
# Recall => True positive rate and sensitivity
# Recall = TP / (TP + FN)
# Precision => Positive predictor value
# Precision = TP / (TP + FP)

# # Random Forest

# In[35]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100) #Create random forest object
rf_model.fit(X_train, y_train.ravel())


# ## Predict Training Data

# In[37]:


rf_predict_train = rf_model.predict(X_train)
#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))


# In[38]:


rf_predict_test = rf_model.predict(X_test)
#testing metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))


# In[39]:


print(metrics.confusion_matrix(y_test, rf_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))


# # Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C=0.7,random_state=42,solver='lbfgs',max_iter=140)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

#testing metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))


# In[61]:


C_start = 0.1
C_end = 5
C_inc = 0.1
C_values, recall_scores = [], []
C_val = C_start
best_recall_score = 0
while (C_val<C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42, solver='lbfgs', max_iter=220)
    lr_model_loop.fit(X_train,y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
    C_val = C_val + C_inc
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(C_values, recall_scores,"-")
plt.xlabel("C value")
plt.ylabel("recall score")


# In[69]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(class_weight="balanced",C=best_score_C_val,random_state=42, solver='lbfgs',max_iter=200)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)
#testing metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
print (metrics.recall_score(y_test,lr_predict_test))


# # LogisiticRegressionCV

# In[70]:


from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1,random_state=42,Cs=3,cv=10,refit=False, class_weight="balanced")
lr_cv_model.fit(X_train, y_train.ravel())


# # Predict on test data

# In[71]:


lr_cv_predict_test = lr_cv_model.predict(X_test)

#testing metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(y_test, lr_cv_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_cv_predict_test))


# In[ ]:




