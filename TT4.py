#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *

# In[2]:


# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# In[3]:


# Load data
train = pd.read_csv('train_fuzz_columns.csv')
test = pd.read_csv('test.csv')
df = train



# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

## Imputation
df[ 'Age' ] = df.Age.fillna( df.Age.mean() )
df[ 'Fare' ] = df.Fare.fillna( df.Fare.mean() )
## filna(-1)

    
## Custom(feature)
title = pd.DataFrame()
title[ 'Title' ] = df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
title[ 'Title' ] = title.Title.map( Title_Dictionary )
df[ 'Title' ] = title[ 'Title' ]
df[ 'Ticket' ] = df[ 'Ticket' ].map( cleanTicket )
df[ 'Cabin' ] = df.Cabin.fillna( 'U' )
df[ 'FamilySize' ] = df[ 'Parch' ] + df[ 'SibSp' ] + 1
df[ 'Family_Single' ] = df[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
df[ 'Family_Small' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
df[ 'Family_Large' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# Basic
# One-hot encoder
cat_feat = ['Title', 'Ticket', 'Cabin'] #   'Ticket', 'Embarked'
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

drop_column = ['Embarked', 'PassengerId', 'Name']
df.drop(drop_column, axis=1, inplace = True)

# Basic
# One-hot encoder
# cat_feat = ['Ticket', 'Cabin'] #   'Ticket', 'Embarked'
# y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

# drop_column = ['Embarked', 'PassengerId', 'Name']
# y1_df.drop(drop_column, axis=1, inplace = True)


