#!/usr/bin/env python
# coding: utf-8

# In[1]:


#You are the Data Scientist at a telecom company “Neo” whose customers are churning out to
#its competitors. You have to analyse the data of your company to find insights to stop your
#customers to churning out to other telecom companies.
#Tasks to be done:
#A) Data Manipulation:
#a. Extract the 5th column & store it in ‘customer_5’
#b. Extract the 15th column & store it in ‘customer_15’
#c. Extract all the male senior citizens whose Payment Method is Electronic check &
#store the result in ‘senior_male_electronic’
#d. Extract all those customers whose tenure is greater than 70 months or their
#Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’
#e. Extract all the customers whose Contract is of two years, payment method is Mailed
#check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
#f. Extract 333 random records from the customer_churndataframe& store the result in
#‘customer_333’
#g. Get the count of different levels from the ‘Churn’ column


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


customer_churn = pd.read_csv('customer_churn.csv')


# In[12]:


customer_churn.head()


# In[13]:


c_15=customer_churn.iloc[:,14]
c_15.head()


# In[14]:


c_random=customer_churn[(customer_churn['gender']=="Male") & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=="Electronic Check")]


# In[15]:


c_random.head()


# In[18]:


c_random=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]


# In[19]:


c_random.head()


# In[22]:


c_random=customer_churn[(customer_churn['Contract']=="Two year") & (customer_churn['PaymentMethod']=="Mailed check") & (customer_churn['Churn']=="Yes")]


# In[23]:


c_random


# In[33]:


c_333=customer_churn.sample(n=333)


# In[34]:


c_333.head()


# In[36]:


customer_churn['Churn'].value_counts()


# In[37]:


customer_churn['Contract'].value_counts()


# In[38]:


#DAta Visulaization


# In[47]:


plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist(),color='Orange')

plt.xlabel("categories of InternetService")
plt.ylabel("coount")
plt.title("Distributions of internet service")


# In[51]:


plt.hist(customer_churn['tenure'],bins=30,color='green')
plt.title("Distribution of Internet service")


# In[57]:


plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title(" Tenure vs Monthly Charges")


# In[58]:


customer_churn.boxplot(column=['tenure'],by=['Contract'])


# In[ ]:


# MAchine learning


# In[60]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y=customer_churn[['MonthlyCharges']]
x=customer_churn[['tenure']]


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# In[62]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[63]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[69]:


y_pred=regressor.predict(x_test)
y_pred[:5],y_test[:5]


# In[68]:


from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,y_pred))


# In[70]:


x=customer_churn[['MonthlyCharges']]
y=customer_churn[['Churn']]


# In[71]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


# In[73]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)


# In[74]:


y_pred=log_model.predict(x_test)


# In[75]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[76]:


confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred)


# In[77]:


(1815)/(1815+651)


# In[80]:


x=customer_churn[['MonthlyCharges','tenure']]
y=customer_churn[['Churn']]


# In[81]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


# In[108]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)


# In[110]:





# In[111]:


confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred)


# In[112]:


x=customer_churn[['tenure']]
y=customer_churn[['Churn']]


from sklearn.tree import DecisionTreeClassifier

my_Tree = DecisionTreeClassifier()
# In[ ]:





# In[ ]:


my_Tree.fit(X_train, y_train


# In[ ]:


y_pred=my_Tree.predict(y_test)


# In[ ]:





# In[98]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[ ]:





# In[ ]:


confusion_matrix(y_test,y_pred),accuracy_score(y_test,y_pred)


# In[100]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)


# In[101]:


rf.predict(x_test)


# In[ ]:





# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:


accuracy_score(y_test,y_pred)

