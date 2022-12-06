#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


import string
def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families


# In[3]:



# Load data
train = pd.read_csv('train_fuzz_columns.csv')
test = pd.read_csv('test.csv')
df = train


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()

## Custom(feature)
df['Embarked'] = df['Embarked'].fillna('S')
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(med_fare)
df['Fare'] = pd.qcut(df['Fare'], 13, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
idx = df[df['Deck'] == 'T'].index
df.loc[idx, 'Deck'] = 'A'
df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')


df['Age'].fillna(df['Age'].median(), inplace = True)
df['Age'] = pd.cut(df['Age'].astype(int), 10)
df['Age'].fillna(df['Age'].mode(), inplace = True)
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df['Family_Size_Grouped'] = df['Family_Size'].map(family_map)
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
df['Family'] = extract_surname(df['Name'])


# ### Encoding
# ## LabelEncoder

non_numeric_features = ['Age', 'Embarked', 'Deck', 'Title', 'Family', 'Family_Size_Grouped']
for feature in non_numeric_features:        
    df[feature] = LabelEncoder().fit_transform(df[feature])

# df.drop(['Cabin'], inplace=True, axis=1)
drop_column = ['PassengerId', 'Cabin', 'Name', 'Ticket']
df.drop(drop_column, axis=1, inplace = True)


# cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


y1_df[ 'Cabin' ] = y1_df.Cabin.fillna( 'U' )
y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df['Embarked'] = y1_df['Embarked'].fillna('S')
non_numeric_features = ['Embarked', 'Cabin']

for feature in non_numeric_features:        
    y1_df[feature] = LabelEncoder().fit_transform(y1_df[feature])
    
    
drop_column = ['PassengerId', 'Name', 'Ticket']
y1_df.drop(drop_column, axis=1, inplace = True)


# In[5]:



seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_titanic_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_titanic_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)
y2_test_orig = y2_data_orig_test.copy()

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)



sc = StandardScaler()
y2_X_train = sc.fit_transform(y2_X_train)
y2_X_test = sc.fit_transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test


y1_X_train = sc.fit_transform(y1_X_train)
y1_X_test = sc.fit_transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test

y2_model = RandomForestClassifier(criterion='gini',
                                       n_estimators=1750,
                                       max_depth=7,
                                       min_samples_split=6,
                                       min_samples_leaf=6,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=42,
                                       n_jobs=-1,
                                       verbose=1) 
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model = RandomForestClassifier(criterion='gini',
                                       n_estimators=1750,
                                       max_depth=7,
                                       min_samples_split=6,
                                       min_samples_leaf=6,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=42,
                                       n_jobs=-1,
                                       verbose=1) 
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)
