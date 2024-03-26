#!/usr/bin/env python
# coding: utf-8

# Business Case : Aerofit - Descriptive Statistics & Probability
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/125/original/aerofit_treadmill.csv?1639992749')


# Analysing Basic Metrics

# Shape Of Data

# In[3]:


df.shape


# In[4]:


df.columns


# Datatypes of Columns

# In[5]:


df.dtypes


# In[6]:


df.index


# In[8]:


df.head(10)


# In[9]:


df.tail(10)


# Missing Value Detection

# In[10]:


np.any(df.isna())


# In[11]:


df.info()


# It can be clearly seen from the above that the DataFrame does not contain any missing value.

# Statistical Summary

# In[12]:


df.describe()


# In[13]:


df.describe(include = object)


# Value counts and unique attributes

# In[14]:


prod_counts = df['Product'].value_counts()
prod_counts


# In[15]:


gender_counts = df['Gender'].value_counts()
gender_counts


# In[16]:


marital_status_counts = df['MaritalStatus'].value_counts()
marital_status_counts


# In[17]:


fitness_counts = df['Fitness'].value_counts()
fitness_counts


# In[18]:


usage_counts = df['Usage'].value_counts()
usage_counts


# In[19]:


df['Education'].value_counts()


# In[20]:


prod_dist = np.round(df['Product'].value_counts(normalize = True) * 100, 2).to_frame()
plt.figure(figsize = (15, 30))
plt.subplot(1, 3, 1)
plt.title('% Contribution of each Product')
plt.pie(x = prod_dist['Product'], explode = [0.005, 0.005, 0.1], labels = prod_dist.index, autopct = '%.2f%%')


gender_dist = (np.round(df['Gender'].value_counts(normalize = True) * 100, 2)).to_frame()
plt.subplot(1, 3, 2)
plt.title('% Contribution of each Gender')
plt.pie(x = gender_dist['Gender'], explode = [0.05, 0], 
        labels = gender_dist.index, autopct = '%.2f%%', colors = ['brown', 'magenta'])


marital_status_dist = (np.round(df['MaritalStatus'].value_counts(normalize = True) * 100, 2)).to_frame()
plt.subplot(1, 3, 3)
plt.title('% Contribution of each Marital Status')
plt.pie(x = marital_status_dist['MaritalStatus'], explode = [0.05, 0], 
        labels = marital_status_dist.index, autopct = '%.2f%%', colors = ['lightblue', 'lightgreen'])
plt.plot()


# Univariate Analysis
# 
# How are the ages of the Aerofit Customers distributed ?

# In[23]:


plt.figure()
sns.histplot(data = df, x = 'Age', kde = True, color = 'magenta')
plt.plot()


# Most of the customers (more than 80% of the total) are aged between 20 and 30 years.
# Less than 10% customers are aged 40 years and above.

# Detecting outliers in age data for aerofit customers

# In[24]:


sns.boxplot(data = df['Age'], width = 0.5, orient = 'h', showmeans = True)
plt.plot()


# Sample Calculation

# In[25]:


result = df[(df["Age"] >= 20) & (df['Age'] <= 35)]['Product'].count() / len(df) * 100
"%% of customers whose age is between 20 and 35 is %.2f%%"%(result)


# In[26]:


data = df['Age']
print('Mean : ', data.mean())
print('Median : ', data.median())
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
print("1st Quartile : ", q1)
print("3rd Quartile : ", q3)
iqr = q3 - q1
print('Innerquartile Range : ', iqr)
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr
print("Upper Bound : ", upper)
print('Lower Bound : ', lower)
outliers = data[(data > upper) | (data < lower)]
print("Outliers : ", sorted(outliers))
len_outliers = len((data[(data > upper) | (data < lower)]))
print('No of Outliers : ', len_outliers)


# Based on the above obtained values, converting age column into bins :

# In[27]:


def age_partitions(x):
    if x <= 24:
        return '<= 24 '
    elif 25 < x <= 33:
        return '25 - 33'
    elif 34 < x <= 46:
        return '34 - 46'
    else:
        return '> 46'
df['age_bins'] = df['Age'].apply(age_partitions)
df['age_bins'].loc[np.random.randint(0, 180, 10)]


# How is the annual income of the Aerofit Customers distributed ?

# In[28]:


plt.figure()
sns.histplot(data = df, x = 'Income', kde = True, bins = 18, color = 'olive')
plt.plot()


# Majority of the customers earn in between 35000 and 60000 dollars annually.
# 80 % of the customers annual salary is less than 65000$.

# Detecting outliers in annual income data of aerofit customers

# In[29]:


plt.figure(figsize = (10, 4))
sns.boxplot(data = df, x = 'Income', width = 0.4, orient = 'h', showmeans = True, fliersize = 3)
plt.plot()


# Sample Calculation :

# In[30]:


data = df['Income']
print('Mean : ', data.mean())
print('Median : ', data.median())
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
print("1st Quartile : ", q1)
print("3rd Quartile : ", q3)
iqr = q3 - q1
print('Innerquartile Range : ', iqr)
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr
print("Upper Bound : ", upper)
print('Lower Bound : ', lower)
outliers = data[(data > upper) | (data < lower)]
print("Outliers : ", sorted(outliers))
len_outliers = len((data[(data > upper) | (data < lower)]))
print('No of Outliers : ', len_outliers)


# Based on the above obtained values, converting age column into bins :

# In[31]:


def income_partitions(x):
    if x < 45000:
        return '< 45k '
    elif 45000 <= x < 60000:
        return '45k - 60k'
    elif 60000 <= x < 80000:
        return '60k - 80k'
    else:
        return '> 80k'
df['income_bins'] = df['Income'].apply(income_partitions)
df['income_bins'].loc[np.random.randint(0, 180, 10)]


# How is the self rated fitness scale of Aerofit Treadmill customers distributed ?

# In[53]:


plt.figure()
sns.histplot(data = df, x = 'Fitness', discrete =True, kde =True, stat ='density', color ='blue')
plt.yticks(np.arange(0,1.0,0.1))
plt.grid(axis ='y')
plt.plot()


# More than 50% customers rate themselves 3 out of 5 in self rated fitness scale
# Around 30% of the total customers rate themselves 4 or above in the fitness scale.
# Around 70 % of the aerofit customers rate themselves 3 or less than 3 in fitness scale.
# Less than 20 % of aerofit customers have excellent shape.

# How is the Education (in years) of Aerofit Treadmill customers distributed ?

# In[54]:


sns.histplot(data = df, x = 'Education', discrete = True, kde = True, color = 'purple')
plt.plot()


# It can be evidently observed in the above plot that most customers have 16 years of Education, followed by 14 years and 18 years.

# How is the number of times the Aerofit Treadmill customers plan to use the treadmill each week distributed ?

# In[56]:


sns.histplot(data = df, x = 'Usage', kde = True, stat = 'density', discrete = True, color = 'green')
plt.plot()


# Based on the above plot, it appears that most customers use treadmills on alternate days.
# There are about 40% of customers who use treadmills three days a week and about 30% who use them four days a week.

# Count of customers vs the expected number of miles customers run / walk each week

# In[57]:


plt.figure()
sns.histplot(data = df, x = 'Miles', kde = True, binwidth = 10, color = 'blue')
plt.plot()


# In[ ]:


On the above plot, we can see that most customers expect to walk or run between 40 and 120 miles a week.


# Bivariate Analysis

# In[58]:


plt.figure(figsize = (12, 8))
plt.title('Count of Customers vs Fitness Scale')
sns.countplot(data = df, x = 'Fitness', hue = 'Gender')                    
plt.grid(axis = 'y')
plt.yticks(np.arange(0, 60, 5))
plt.ylabel('Count of Customers')
plt.plot()


# Most of the males and females (more than 50% customers) find themselves in the fitness scale 3 .
# There is a slight difference in the number of males and females in all the fitness scales except for high fitness scales.
# For fitness scales 4 and 5, there are roughly 3 times more males than females.

# In[59]:


plt.figure(figsize = (10, 6))
plt.title("Products' Count")
sns.countplot(data = df, x = 'Product', hue = 'Gender')
plt.grid(axis = 'y')
plt.plot()


# It can be observed that most people buy the entry-level treadmills.
# The number of males buying the treadmills having advanced features is around 5 times the number of females buying the same.

# In[60]:


# For Male, different product categories and 
plt.figure(figsize = (12, 8))
plt.title("Count of Customers vs Product type")
plt.yticks(np.arange(0, 60, 5))
sns.countplot(data = df, x = 'Product', hue = 'Fitness')
plt.ylabel('Count of Customers')
plt.grid(axis = 'y')
plt.plot()


# The customers who rate themselves 3 out of 5 in self rated fitness scale are more likely to invest in the entry-level treadmills or treadmills for mid-level runners i.e., KP281 and KP481 respectively and they are more unlikey to buy the treadmill which has advanced features i.e., KP781.
# 
# The treadmill having advanced features are mostly used by the people with high fitness levels.
# 
# The customers who rate themselves 3 or below in the self-rated fitness scale do not buy KP781.

# In[61]:


plt.figure(figsize = (12, 8))
sns.countplot(data = df, x = 'age_bins', hue = 'Product')
plt.grid(axis = 'y')
plt.plot()


# In[62]:


plt.figure(figsize = (12, 8))
sns.scatterplot(data = df, x= 'Age', y = 'Income', hue = 'Product', size = 'Fitness')
plt.show()


# The customers having high annual income and high fitness scale generally buys KP781.
# The customers having low fitness scale or low annual income generally buy KP281 and KP481.

# What is the age range of the customers who purchase a specific type of product?

# In[63]:


plt.figure(figsize = (12, 8))
sns.boxplot(data = df, x = 'Product', y = 'Age', hue = 'Gender', showmeans = True)
plt.plot()


# In[ ]:


Most customers were in their 20s or 30s.
The age range of KP781 customers is smaller than the age range of the customers who bought other two products.
There is a significant difference in the median age of males and females who bought KP481.
For any product, the age range for males is higher than that of female. The range difference is significant for the product KP781.


# Sample calculation to detect outliers in the age of males who bought KP781

# In[64]:


data = df.loc[(df['Product'] == 'KP781') & (df['Gender'] == 'Male'), 'Age']
print('Mean : ', data.mean())
print('Median : ', data.median())
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
print("Quartile 1 : ", q1)
print("Quartile 3 : ", q3)
iqr = q3 - q1
print('Inner Quartile Range : ', iqr)
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr
print("Upper : ", upper)
print('Lower : ', lower)
outliers = data[(data > upper) | (data < lower)]
print("Outliers : ", list(outliers))
len_outliers = len((data[(data > upper) | (data < lower)]))
print('No of Outliers : ', len_outliers)


# In[ ]:


We can clearly see in the boxplot above the sample calculation that we have exactly 4 outliers in the data of age of the males who bought KP781 treadmill.


# What is the income range of the customers who purchase a specific type of product?

# In[65]:


plt.figure(figsize = (12, 8))
sns.boxplot(data = df, x = 'Product', y = 'Income', hue = 'Gender', showmeans = True, fliersize = 4)
plt.plot()


# The median income of customers who bought KP781 is much higher than that of the customers who bought other two products.
# The range of income for customers buying KP781 is much higher than the same for customers buying KP281 and KP481.

# Sample calculation to detect outliers in the income of females who bought KP481

# In[66]:


data = df.loc[(df['Product'] == 'KP481') & (df['Gender'] == 'Female'), 'Income']
print('Mean : ', data.mean())
print('Median : ', data.median())
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
print("Quartile 1 : ", q1)
print("Quartile 3 : ", q3)
iqr = q3 - q1
print('Inner Quartile Range : ', iqr)
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr
print("Upper : ", upper)
print('Lower : ', lower)
outliers = data[(data > upper) | (data < lower)]
print("Outliers : ", list(outliers))
len_outliers = len((data[(data > upper) | (data < lower)]))
print('No of Outliers : ', len_outliers)


# In[67]:


plt.figure(figsize = (12, 8))
sns.countplot(data = df, x = 'income_bins', hue = 'Product')
plt.grid(axis = 'y')
plt.plot()


# The customers with high annual salary (60k and above) are more likely to buy KP781.
# The customers with annual salary < 60k are more likely to buy KP281 and KP481.

# Coorelation between measurable quantities

# In[68]:


sns.pairplot(data = df, kind = 'reg') 
plt.plot()


# In[69]:


df_corr = df.corr()
df_corr


# In[70]:


plt.figure(figsize = (12, 8))
sns.heatmap(data = df_corr, 
            annot = True, 
            fmt = '.2%', 
            cmap='Greens', 
            linewidth = 2, 
            linecolor = 'black', 
            annot_kws = {'fontsize' : 'large',
                        'fontfamily' : 'serif',
                        'fontweight': 'bold'})           
plt.plot()


# The customer with high fitness scale is more likely to run or walk more miles.
# The customer who expects to use the treadmill more times in a week generally expects to walk or run more miles in the week.
# The customer who have a high fitness scale generally uses the treadmill more frequently in a week.

# What is the product buying behaviors of both the genders ?

# In[71]:


print(pd.crosstab(index = df['Product'], columns = df['Gender'], margins = True))
print()
print('-' * 26)
print()
print("Product-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['Gender'], normalize = 'index') * 100, 2))
print()
print('-' * 23)
print()
print("Gender-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['Gender'], normalize = 'columns') * 100, 2))


# Customers who bought KP781, 82.5% of them are males rest are females.
# 
# Among all female customers, only 9.21 % buy KP781. Females mostly buy products KP281 or KP481.

# What is the probability of buying a specific product provided the customer is of specific gender ?

# In[72]:


products = df['Product'].unique()
genders = df['Gender'].unique()
for i in genders:
    for j in products:
        prob = len(df[(df['Gender'] == i) & (df['Product'] == j)]) / len(df[df['Gender'] == i])
        prob = np.round(prob * 100, 2)
        print("Probability of buying '{}' provided the customer is {} is {}% ".format(j, i, prob))
        print()


# What is the probability of that the customer is of specific gender provided specific product is bought ?

# In[73]:


products = df['Product'].unique()
genders = df['Gender'].unique()
for i in genders:
    for j in products:
        prob = len(df[(df['Gender'] == i) & (df['Product'] == j)]) / len(df[df['Product'] == j])
        prob = np.round(prob * 100, 2)
        print("Probability that the customer is {} provided {} was bought is {}% ".format(i, j, prob))
        print()


# What is the product buying behaviors of both the Marital Statuses ?

# In[74]:


print(pd.crosstab(index = df['Product'], columns = df['MaritalStatus'], margins = True))
print()
print('-' * 37)
print()
print("Product-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['MaritalStatus'], normalize = 'index') * 100, 2))
print()
print('-' * 33)
print()
print("Marital Status-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['MaritalStatus'], normalize = 'columns') * 100, 2))


# What is the probability of buying a specific product provided the customer is of specific marital status ?

# In[75]:


products = df['Product'].unique()
statuses = df['MaritalStatus'].unique()
for i in statuses:
    if i != 'Single':
        print('-' * 76)
    for j in products:
        prob = len(df[(df['MaritalStatus'] == i) & (df['Product'] == j)]) / len(df[df['MaritalStatus'] == i])
        prob = np.round(prob * 100, 2)
        print("Probability of buying '{}' provided the customer is '{}' is {}% ".format(j, i, prob))
        print()


# What is the probability of that the customer is of specific Marital Status provided specific product is bought ?

# In[76]:


products = df['Product'].unique()
statuses = df['MaritalStatus'].unique()
for i in statuses:
    if i != 'Single':
        print('-' * 82)
    for j in products:
        prob = len(df[(df['MaritalStatus'] == i) & (df['Product'] == j)]) / len(df[df['Product'] == j])
        prob = np.round(prob * 100, 2)
        print("Probability that the customer is '{}' provided '{}' was bought is {}% ".format(i, j, prob))
        print()


# What is the product buying behaviors of customers with different fitness levels ?

# In[77]:


print(pd.crosstab(index = df['Product'], columns = df['Fitness'], margins = True))
print()
print('-' * 40)
print()
print("Product-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['Fitness'], normalize = 'index') * 100, 2))
print()
print('-' * 40)
print()
print("Fitness Scale-wise normalization : ")
print(np.round(pd.crosstab(index = df['Product'], columns = df['Fitness'], normalize = 'columns') * 100, 2))


# Number of customers who bought products KP281, KP481 and KP781 are in ratio 4 : 3 : 2. That means for every 9 customers, 4 customers bought KP281, 3 bought KP481 and 2 bought KP781.
# 
# Among all the customers who bought KP281, 96.25 % of them had fitness scales of 2, 3 or 4. Only 2.5 % of them had excellent body shape.
# 
# Among all the customers who bought KP781, 90 % of them had fitness scales 4 or 5. Only 10 % of them had average body shape.
# 
# Among all the customers who had excellent body shape (fitness scale 5), 93.55 % of them bought product KP781 while the rest buy KP281.
# 
# All the customers in each fitness levels 1 and 2 (i.e., customers having poor body shape) either bought product KP281 or KP481. None of them bought the treadmill having advanced features i.e., KP781.

# What is the probability of buying a specific product provided the customer has specific fitness scale ?

# In[78]:


products = df['Product'].unique()
scales = sorted(df['Fitness'].unique())
for i in scales:
    if i != 1:
        print('-' * 88)
    for j in products:
        prob = len(df[(df['Fitness'] == i) & (df['Product'] == j)]) / len(df[df['Fitness'] == i])
        prob = np.round(prob * 100, 2)
        print("Probability of buying '{}' provided the customer has the fitness scale '{}' is {}% ".format(j, i, prob))
        print()


# What is the probability of that the customer has a specific fitness scale provided specific product was bought ?

# In[79]:


products = df['Product'].unique()
scales = sorted(df['Fitness'].unique())
for i in scales:
    if i != 1:
        print('-' * 94)
    for j in products:
        prob = len(df[(df['Fitness'] == i) & (df['Product'] == j)]) / len(df[df['Product'] == j])
        prob = np.round(prob * 100, 2)
        print("Probability that the customer has a fitness scale of '{}' provided '{}' was bought is {}% ".format(i, j, prob))
        print()


# What is the relation between Marital Statuses and fitness levels of the Aerofit Customers?

# In[80]:


print(pd.crosstab(index = df['MaritalStatus'], columns = df['Fitness'], margins = True))
print()
print('-' * 48)
print('Marital Status wise normalization : ')
print()
print(np.round(pd.crosstab(index = df['MaritalStatus'], columns = df['Fitness'], normalize = 'index') * 100, 2))
print()
print("-" * 48)
print('Fitness levels wise normalization : ')
print()
print(np.round(pd.crosstab(index = df['MaritalStatus'], columns = df['Fitness'], normalize = 'columns') * 100, 2))


# Majority of customers (i.e., greater than 50%) in each marital statuses had fitness scale 3.
# Majority of customers (i.e., greater than 50%) in each of fitness scales 2, 3, 4 and 5 were partnered.(Since there are significantly higher number of customers who were partnered than single)

# What is the relation between Incomes and Products bought by the Aerofit Customers?

# In[81]:


print(pd.crosstab(index = df['Product'], columns = df['income_bins'], margins = True))
print()
print('-' * 54)
print('Product wise normalization : ')
print()
print(np.round(pd.crosstab(index = df['Product'], columns = df['income_bins'], normalize = 'index') * 100, 2))
print()
print("-" * 48)
print('Income-bins wise normalization :')
print()
print(np.round(pd.crosstab(index = df['Product'], columns = df['income_bins'], normalize = 'columns') * 100, 2))


# What is the probability of buying a specific product provided the customer's annual income lies in a specific income range ?

# In[82]:


products = df['Product'].unique()
incomes = sorted(df['income_bins'].unique())
for i in incomes:
    if i != '45k - 60k':
        print('-' * 105)
    for j in products:
        prob = len(df[(df['income_bins'] == i) & (df['Product'] == j)]) / len(df[df['income_bins'] == i])
        prob = np.round(prob * 100, 2)
        print("Probability of buying '{}' provided the customer has the annual income in range '{}' is {}% ".format(j, i, prob))
        print()


# What is the probability of that the customer's annual income lies in a specific salary range provided specific product was bought ?

# In[83]:


products = df['Product'].unique()
incomes = sorted(df['income_bins'].unique())
for i in incomes:
    if i != '45k - 60k':
        print('-' * 105)
    for j in products:
        prob = len(df[(df['income_bins'] == i) & (df['Product'] == j)]) / len(df[df['Product'] == j])
        prob = np.round(prob * 100, 2)
        print("Probability that the customer's annual income lies in range '{}' provided '{}' was bought is {}% ".format(i, j, prob))
        print()


# What is the relation between Age Categories and Products bought by the Aerofit Customers?

# In[84]:


print(pd.crosstab(index = df['Product'], columns = df['age_bins'], margins = True))
print()
print('-' * 45)
print('Product wise normalization : ')
print()
print(np.round(pd.crosstab(index = df['Product'], columns = df['age_bins'], normalize = 'index') * 100, 2))
print()
print("-" * 42)
print('Age-bins wise normalization : ')
print()
print(np.round(pd.crosstab(index = df['Product'], columns = df['age_bins'], normalize = 'columns') * 100, 2))


# What is the probability of buying a specific product provided the customer's age lies in a specific age range ?

# In[85]:


products = df['Product'].unique()
ages = sorted(df['age_bins'].unique())
for i in ages:
    if i != '25 - 33':
        print('-' * 91)
    for j in products:
        prob = len(df[(df['age_bins'] == i) & (df['Product'] == j)]) / len(df[df['age_bins'] == i])
        prob = np.round(prob * 100, 2)
        print("Probability of buying '{}' provided the customer's age lies in range '{}' is {}% ".format(j, i, prob))
        print()


# What is the probability of that the customer's age lies in a specific age range provided specific product was bought ?

# In[86]:


products = df['Product'].unique()
ages = sorted(df['age_bins'].unique())
for i in ages:
    if i != '25 - 33':
        print('-' * 96)
    for j in products:
        prob = len(df[(df['age_bins'] == i) & (df['Product'] == j)]) / len(df[df['Product'] == j])
        prob = np.round(prob * 100, 2)
        print("Probability that the customer's age lies in range '{}' provided '{}' was bought is {}% ".format(i, j, prob))
        print()


# Customer Profiling :
# Product of buying a specific product based on gender, age, fitness scale, income :

# In[95]:


products = df['Product'].unique()
genders = df['Gender'].unique()
ages = df['age_bins'].unique()
fitnesses = sorted(df["Fitness"].unique())
statuses = df['MaritalStatus'].unique()
incomes = df['income_bins'].unique()
for i in products:
    for j in genders:
        for k in statuses:
            for l in ages:
                for m in fitnesses:
                    for n in incomes:
                        try : 
                            count += 1
                            res = np.round(len(df[df['Product'] == i]) / len(df[(df['Gender'] == j) & (df['age_bins'] == l) & (df['Fitness'] == m) & (df['MaritalStatus'] == k) & (df['income_bins'] == n)]), 2)
                            print("P({} / ({}, {}, age {}, fitness scale = {}, income {})) = {}%".format(i, j, k, l, m, n, res))
                        except:
                            print("No record for ({}, {}, age {}, fitness scale = {}, income {}) buying {}".format(j, k, l, m,n,i))


# Insights

# Number of customers who bought products KP281, KP481 and KP781 are in ratio 4 : 3 : 2. That means for every 9 customers, 4 customers bought KP281, 3 bought KP481 and 2 bought KP781.
# There are more male customers than females. Around 60% of the total customers are males.
# There are more customers who are partnered than single. Almost 60% of customers are partnered.
# Age of the customers varies between 18 and 50 years.
# More than 80% of the total customers are aged between 20 and 30 years.
# Annual income of the customers varies in the range of 29562 dollars to 104581 dollars.
# 80 % of the customers annual salary is less than 65000 dollars.
# Expected usage of treadmills lies in the range of 2 to 7 times in a week.
# Expected number of miles that the customer walks or runs vary between 21 miles to 360 miles per week.
# More than 50% customers rate themselves 3 out of 5 in self rated fitness scale
# Around 70 % of the aerofit customers rate themselves 3 or less in fitness scale.
# There are about 40% of customers who use treadmills three days a week and about 30% who use them four days a week.
# For fitness scales 4 and 5, there are 3 times more males than females.
# Among all the customers who bought KP781, 90 % of them had fitness scales 4 or 5. Only 10 % of them had average body shape.
# The number of males buying the treadmills having advanced features is around 5 times the number of females buying the same.
# The treadmill having advanced features are mostly bought by the people with high fitness levels.
# The customers having high annual income (> 60k dollars) and high fitness scales(> 4) generally buy KP781.
# The customers who rate themselves 1 or 2 in the self-rated fitness scale do not buy KP781.
# Customers who bought KP781, 82.5% of them are males rest are females.
# Among all female customers, only 9.21 % buy KP781. Females mostly buy products KP281 or KP481.
# Among all the customers who bought KP281, 96.25 % of them had fitness scales of 2, 3 or 4. Only 2.5 % of them had excellent body shape.
# Among all the customers who had excellent body shape (fitness scale 5), 93.55 % of them bought product KP781 while the rest buy KP281.
# All the customers in each fitness levels 1 and 2 (i.e., customers having poor body shape) either bought product KP281 or KP481. None of them bought the treadmill having advanced features i.e., KP781.
# Probability of buying 'KP781' provided the customer has the annual income in range '> 80k' is 100.0%.

# Recommendations :

# Recommend KP781 product to users who exercises/run more frequently and run more and more miles , and have high income. Since Kp781 is least selling product (22.2% share of all the products) , recommend this product some customers who exercise at intermediate to extensive level , if they are planning to go for KP481. Also the targeted Age Category is Adult and age above 45.
# Recommend KP481 product specifically for female customers who run/walk more miles , as data shows their probability is higher. Statistical Summery about fitness level and miles for KP481 is not good as KP281 which is cheaper product. Possibly because of price, customers prefer to purchase KP281. It is recommended to make some necessary changes to product K481 to increase customer experience.
