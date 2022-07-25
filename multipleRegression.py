import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

class SMWrapper(BaseEstimator, RegressorMixin):
    """
    This is a class which will help run multivariate regression
    Built online from... Not built by hand (so we are basically using this
    class in the same sense that we use packages)
    https://stackoverflow.com/questions/41045752/using-statsmodel
    -estimations-with-scikit-learn-cross-validation-is-it-possible
    ...

    Attributes
    ----------
    model_class : sklearn.model_selection
        a multivariate model
    fit_intercept : array
        gives a list of intercepts

    Methods
    -------
    fit(self, X, y):
        used to find the betas 

    predict(self, X):
        used to predict on a test set 

                
    """    
    def __init__(self, model_class, fit_intercept=True):
        """
        Constructs class object
        @param model_class: type of model (2,3,4...,n features)
        @param fit_intercept: intercept for regression equation
        """
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        """
        Fits class object to training set
        @param X: a numpy array of inputs
        @param y: outputs

        """
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        """
        Gives pre
        @param X: gives predictions based on inputs
        """
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


def normalize(df,colName):
    '''
    takes in pandas column... Shifts values so that mean is 0.
    @param df: A dataframe containing the rows and columns that will be normalized.
    @param colName: A string containing the name of the column that will be normalized.
    @return: A dataframe that has had its mean shifted to 0.
    '''
    mean = df[colName].sum() / len(df[colName])
    avg_array = np.array([mean]*len(df[colName]))
    df[colName] = df[colName] - avg_array
    df[colName] = df[colName] / df[colName].std()
    return df

