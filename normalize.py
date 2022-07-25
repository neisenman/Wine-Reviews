import numpy as np
import pandas as pd

def normalize(df,colName):
    '''
    takes in pandas column and shifts values of a column
    so that mean is 0.
    @param df: pandas dataframe
    @param colName: name of column you wish to normalize
    @return dataframe with a single normalized columns
    '''
    
    mean = df[colName].sum() / len(df[colName])
    avg_array = np.array([mean]*len(df[colName]))
    df[colName] = df[colName] - avg_array
    df[colName] = df[colName] / df[colName].std()
    return df

def calcError(y_pred, y):
    '''
    Calculates residual sum of squares given a predicted series
    an output series
    @param y_pred: predicted values for the output
    @param y: actual values for output
    @return the residual sum of squares
    '''
    return np.sum((y_pred-y)**2)/len(y_pred)
