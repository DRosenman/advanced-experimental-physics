# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:16:45 2017

@author: Dave Rosenman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def slope(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        return ((x-x.mean())*y).sum()/((x-x.mean())**2).sum()
    else:
        return ((x*y).sum())/(x**2).sum()

def intercept(x,y,through_origin = False):
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        return y.mean() - slope(x,y)*x.mean()
    else:
        return 0.0

def slope_error(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    if through_origin == False:
        residuals_squared = (y - slope(x,y)*x - intercept(x,y))**2
        residuals_squared_sum = residuals_squared.sum()
        D = ((x-x.mean())**2).sum()
        return np.sqrt((1/(n-2))*residuals_squared_sum/D)
    else:
        residuals_squared = (y - slope(x,y,True)*x)**2
        return np.sqrt(residuals_squared.sum()/((n-1)*(x**2).sum()))

def intercept_error(x,y,through_origin = False):
    if through_origin == False:
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        D = ((x-x.mean())**2).sum()
        residuals_squared = (y - slope(x,y)*x - intercept(x,y))**2
        return np.sqrt(((1/n) + (x.mean()**2)/D)*residuals_squared.sum()/(n-2))
    else:
        return 'NaN'

def summary_stats(x,y,through_origin = False):
    return (slope(x,y,through_origin),intercept(x,y,through_origin),
            slope_error(x,y,through_origin),intercept_error(x,y,through_origin),
                       correlation_coefficient(x,y,through_origin))

def summary_stats_df(x,y,both = False,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if both == False:
        if through_origin == False:
            df = pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','Std Error, Intercept','R'],
                                       index = [''])
            return df
        else:
            df = pd.DataFrame([[slope(x,y,through_origin),intercept(x,y,through_origin = True),slope_error(x,y,through_origin),correlation_coefficient(x,y,through_origin)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope','R'],
                                       index = [''])
            return df
    if both == True:
        df = pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y),intercept_error(x,y),correlation_coefficient(x,y)],
                                [slope(x,y,True),intercept(x,y,True),slope_error(x,y,True),'NA',correlation_coefficient(x,y,True)]],
                                columns = ['Slope', 'Intercept','Std. Error, Slope','Std Error, Intercept','R'],
                                index = ['Regular Linear Regression','Regression Line Through (0,0)'])
        return df

def print_summary_stats(x,y,through_origin = False):
    if through_origin == False:
        print(pd.DataFrame([[slope(x,y),intercept(x,y),slope_error(x,y)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope'],
                                       index = ['']))
        print("")
        print(pd.DataFrame([[intercept_error(x,y),correlation_coefficient(x,y)]],
                            columns = ['Std. Error, Intercept', 'r'],
                            index = ['']))
    if through_origin == True:
        print(pd.DataFrame([[slope(x,y,through_origin = True),intercept(x,y,through_origin = True),
                            slope_error(x,y,through_origin = True)]],
                            columns = ['Slope', 'Intercept','Std. Error, Slope'],
                            index = ['']))
        print("")

        print(pd.DataFrame([[correlation_coefficient(x,y,through_origin = True)]],columns = ['r'], index = ['']))

def correlation_coefficient(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == False:
        n = len(x)
        xy = x*y
        return (n*(xy.sum()) - (x.sum())*(y.sum()))/((((n*(x**2).sum()) - (x.sum())**2)**.5)*((n*((y**2).sum()) - (y.sum())**2)**.5))
    else:
        return np.sqrt(1.0 - (((y - x*slope(x,y,through_origin))**2).sum())/((y**2).sum()))


def r(x,y,through_origin = False):
    return correlation_coefficient(x,y,through_origin)

def r_squared(x,y,through_origin = False):
    return (correlation_coefficient(x,y,through_origin))**2

import scipy.stats as stats

def p_value(x,y,through_origin = False):
    x = np.array(x)
    y = np.array(y)
    if through_origin == True:
        k = 1
    else:
        k = 2
    df = len(x) - k
    m = slope(x,y,through_origin)
    error = slope_error(x,y,through_origin)
    t = m/error
    return stats.t.sf(np.abs(t), df)*2
