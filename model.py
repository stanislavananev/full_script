from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from catboost import CatBoostRegressor
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def my_custom_scorer(y_true, y_pred): #5% deviation
    diff = np.abs(y_true - y_pred)
    mask = np.int32( diff>0.05*y_true )
    res = ( len(y_true)-np.count_nonzero(mask) )/len(y_true)
    return res

scorer = make_scorer(my_custom_scorer, greater_is_better=True)

params = {
    'DecisionTreeRegressor': {'max_depth': [5,10,15,20,30,40,50,60,80,100,120,140]
                              , 'min_samples_split' : [2,3,4,5]},
#     'SVR': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2,3,4,5]},
#     'KNeighborsRegressor': {'n_neighbors': [3,4,5,6,7], 'weights':['uniform', 'distance']},
    
    'GradientBoostingRegressor': {

          'learning_rate': [0.05]#[0.1,0.01]
         , 'max_depth': [3,4,5,6,10,15,20]
         , 'min_samples_split' : [2,3,4,5]
         , 'n_estimators': list(range(10,105,10))
         , 'loss': ['absolute_error', 'huber']
         , 'criterion': ['friedman_mse', 'squared_error']
         
    
    },
    
    'CatBoostRegressor': {
            'depth': [2,4,6,10,15]
         , 'learning_rate': [0.05,0.1]
         , 'iterations': [10, 30, 50, 100]
         , 'verbose': [False]
         , 'loss_function':['MAE', 'RMSE']
    }
    
}


# Class for predictive model
class super_model:
    def __init__(self,  models=[CatBoostRegressor], scoring=my_custom_scorer, cv=5, max_models=12):   
        super().__init__()
        self.models = models
        self.model_names = [type(model()).__name__ for model in models]
        self.scoring = scoring
        self.cv = cv
        self.workers = []
        self.weights = []
        self.max_models = max_models
        self.extrapolator = LinearRegression(positive=True)
        self.pt_bounds = (None, None)
        
    def fit(self, data):
        X, y = data.iloc[:, :-1].astype(np.float64), data.iloc[:, -1].astype(np.float64)
        
        Pt = np.array( X['Pt'] )
        y -= Pt
        assert (y>=0).all()
        self.pt_bounds = (min(Pt), max(Pt))
        self.extrapolator.fit(X,y)


        scores = []
        best_params = []
        workers = []
        scorer = make_scorer(self.scoring, greater_is_better=True)
        for i in range( len(self.models) ):
            grs = GridSearchCV(
                                 estimator=self.models[i]()
                               , param_grid=params[self.model_names[i]]
                               , cv=self.cv
                               , scoring=scorer
                               , n_jobs=-1)
            grs.fit(X,y)
            
            s = np.array(grs.cv_results_['mean_test_score'])
            pars = np.array(grs.cv_results_['params'])
            pars = list( pars[~np.isnan(s)] )
            s = list( s[~np.isnan(s)] )
            

            scores += s
            best_params += pars
            workers += [self.models[i]( **p ) for p in pars]
            

        score_threshold = 0.97*np.max(scores)
        workers = [workers[i] for i in range(len(workers)) if scores[i]>=score_threshold]
        scores = [scores[i] for i in range(len(scores)) if scores[i]>=score_threshold]
        
        if len(scores)>self.max_models:
            ind = np.argpartition(scores, -self.max_models)[-self.max_models:]
            workers = np.array(workers)[ind]
            scores = np.array(scores)[ind]
        self.weights = scores / np.sum(scores)
        self.workers = workers
        for i in range(len(self.workers)):
            self.workers[i].fit(X, y)
            
        self.score = np.mean( scores  )
        
    def predict(self, x):
        X = [x]
        if self.pt_bounds[0]<=x[-1]<=self.pt_bounds[1]:
            predictions = [self.workers[i].predict(X)*self.weights[i] for i in range(len(self.workers))]
            y_pred = np.sum(predictions)
        else:
            y_pred = self.extrapolator.predict(X)
        y_pred += x[-1]
        return y_pred
        
    def evaluate(self, X, y):
        predictions = []
        for i in range(len(self.workers)):
            predictions.append(self.workers[i].predict(X)*self.weights[i])
        y_pred = np.sum(predictions, axis=0)
        y -= X.iloc[:, -1]
        return self.scoring(y, y_pred)
    
    
    
    
    


    
    