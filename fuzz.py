import pandas as pd
import numpy as np

def addNans(df, amount: float):
    # Add random nans to the dataframe based on the given percentage amount
    # amount: float between 0 and 1
    # df: pandas dataframe
    # return: pandas dataframe with nans added
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].mask(np.random.random(len(df)) < amount)
    return df

def addNansToColumns(df, amount: float):
    # Add random nans to the every column in dataframe based on the given percentage amount
    # amount: float between 0 and 1
    # df: pandas dataframe
    # return: pandas dataframe with nans added
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].mask(np.random.random(len(df)) < amount)
    return df

def addGarbage(df, amount: float):
    # Add random numbers or letters to the dataframe based on the given percentage amount
    # amount: float between 0 and 1
    # df: pandas dataframe
    # return: pandas dataframe with garbage added
    df = df.copy()
    
    #for each column, if the column is numeric or not
    for col in df.columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            #if numeric, add random numbers
            df[col] = df[col].mask(np.random.random(len(df)) < amount, np.random.random(len(df)))
        else:
            #if not numeric, add random letters
            df[col] = df[col].mask(np.random.random(len(df)) < amount, np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), len(df)))
    return df

def changeTimeFormat(df, format):
    # Change the time format of the dataframe
    # format: string of the format to change to
    # df: pandas dataframe
    # return: pandas dataframe with time format changed
    df = df.copy()
    
    #for each column, if the column is a time column
    for col in df.columns:
        if df[col].dtype == np.datetime64 or df[col].dtype == np.timedelta64:
            #change the format
            df[col] = pd.to_datetime(df[col], format=format)
    return df

def changeTimeFormatToISO(df):
    # Change the time format of the dataframe to ISO using the changeTimeFormat function
    # df: pandas dataframe
    # return: pandas dataframe with time format changed
    df = df.copy()

    #use changeTimeFormat function to change to ISO
    return changeTimeFormat(df, '%Y-%m-%dT%H:%M:%S.%f')

def changeTimeFormatToUnix(df):
    # Change the time format of the dataframe to Unix using the changeTimeFormat function
    # df: pandas dataframe
    # return: pandas dataframe with time format changed
    df = df.copy()

    #use changeTimeFormat function to change to Unix
    return changeTimeFormat(df, '%s')

def changeTimeFormatToUnixMs(df):
    # Change the time format of the dataframe to UnixMs using the changeTimeFormat function
    # df: pandas dataframe
    # return: pandas dataframe with time format changed
    df = df.copy()

    #use changeTimeFormat function to change to UnixMs
    return changeTimeFormat(df, '%s.%f')

def addGarbageToTime(df, amount: float):
    # Add random numbers or letters to the time columns in the dataframe based on the given percentage amount
    # amount: float between 0 and 1
    # df: pandas dataframe
    # return: pandas dataframe with garbage added
    df = df.copy()
    
    #for each column, if the column is a time column
    for col in df.columns:
        if df[col].dtype == np.datetime64 or df[col].dtype == np.timedelta64:
            #if time column, add random numbers
            df[col] = df[col].mask(np.random.random(len(df)) < amount, np.random.random(len(df)))
    return df