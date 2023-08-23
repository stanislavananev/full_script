from sklearn.metrics import mean_absolute_percentage_error as mape
import pickle
import numpy as np
import pandas as pd

# 5% deviation
def my_custom_scorer(y_true, y_pred): 
    diff = np.abs(y_true - y_pred)
    mask = np.int32( diff>0.05*y_true )
    res = ( len(y_true)-np.count_nonzero(mask) )/len(y_true)
    return res

# Односторонний скор
def s(model1,data1, model2):
    X, y_true = data1.iloc[:,:-1], data1.iloc[:,-1]
    y_pred = [model2.predict(x) for x in np.array(X)]
    return my_custom_scorer(y_true, y_pred)/model1.score

# Двусторонний скор
def s2(model1,data1, model2, data2):
    return (s(model1, data1, model2)+s(model2, data2, model1))/2 

# mean_absolute_percentage_error
def MAPE(model, data):
    X, y_true = data1.iloc[:,:-1], data1.iloc[:,-1]
    y_pred = [model2.predict(x) for x in np.array(X)]
    return mape(y_true, y_pred)

# find score for analoges
def similarities(well, gen):
    model1 = pickle.load(open(f'models/{well}.pickle', "rb"))
    data1 = gen[well].dropna()

    res = []
    S = []
    for well2 in gen.mask_indices:
        if well2 != well:
            model2 = pickle.load(open(f'models/{well2}.pickle', "rb"))
            S.append( (well2, s(model1, data1, model2)) )
            
    return S