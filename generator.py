# Generator class

from datetime import datetime
import os
import numpy as np
import pandas as pd
from collections import Counter
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import re

import warnings
warnings.filterwarnings("ignore")


def filter_data(data):
    cond1 = (data['G']<=3000) | (data['G'].isna()) #&(data['Pb']-data['Pt']<10)
    cond2 = (data['Q']<=200) | (data['Q'].isna())
    cond3 = (data['Pb']<=200) | (data['Pb'].isna())
    cond4 = (data['W']<=100) | (data['W'].isna())
    cond5 = (data['Pt']<=200) | (data['Pt'].isna())
    cond6 = (data['Pt']<=(data['Pb'])) | (data['Pt'].isna()) | (data['Pb'].isna())
    return data[cond1 & cond2 & cond3 & cond4 & cond5 & cond6].dropna().drop_duplicates()

class Snapshot_Generator:
    def __init__(self, data=None
                 , path=None
                 , skip_rows=0
                 , mode='dynamic'
                 , interpolate = None
                 , filtering = False):
         
        if data is not None:
            self.data = data
        elif path is not None:
            files = os.listdir(path)
            files.sort(key=lambda x: x.split('.')[-2])
            #print(files)
            data = [self.excel_to_df(path+f'/{file_name}', skip_rows)
                                    for file_name in files  
                                                        if '.~lock.' not in file_name and '.xls' in file_name]       
            self.data = pd.concat(data, axis=1).T.drop_duplicates().T
        else:
            raise Exception('No data provided!')
        self.indices = self.data.index.drop_duplicates().dropna()
        self.index = 0
        self.interpolate=interpolate
        self.filtering = filtering
        
        
        self.deparol_dict = dict()

        # Считываем ВСЕ скважины в правильном порядке для составления масок
        file = open('/Users/stanislavananyev/PycharmProjects/GPN/meta_well-main/Wells.txt', 'r')
        Lines = file.readlines()
        code = 1000
        # Создаем маски

        pattern = r"\d{3}"
        for line in Lines:
            line = re.findall(pattern, line)[0]
            self.deparol_dict[line] = str(code)
            code += 1
            if code > 1108:
                break
            
        self.indices = [ind for ind in self.indices if re.findall(pattern, ind)[0] in list(self.deparol_dict.keys())]
        self.n = len(self.indices)
        tmp = [re.findall(pattern, ind)[0] for ind in self.indices]
        # Обратный словарь соответствий маска - скважина 
        self.mask_indices = [self.deparol_dict[ind] for ind in tmp]
        if mode not in ['static', 'dynamic']:
            raise Exception('Unknown mode! Please, choose "static" or "dynamic"')
        else:
            self.mode = mode
     
    def filter_wells(self, wells):
        assert len(wells)>0
        # Обратный словарь соответствий маска - скважина 
        inv_map = {v: k for k, v in self.deparol_dict.items()}
        wells = [inv_map[str(well)] for well in wells]
        self.indices = [well for well in self.indices if well[:3] in wells]
        self.n = len(self.indices)
        tmp = [ind[:3] for ind in self.indices]
        self.mask_indices = [self.deparol_dict[ind] for ind in tmp]
    def interpolate_data(self, data, column):
        features = [col for col in data.columns if col != column]
        if len(data[column].dropna()) == 0:
            return data
        
        tmp = data.isna().sum(axis=1)
        mask = tmp[tmp==0].index
        train = data.loc[mask]
        if len(train) == 0:
            return data
        

        cols = [col for col in  train.columns if col != column]
        model = DecisionTreeRegressor(max_depth=500, random_state=0)#n_estimators=500)
        X, y = train.loc[:,cols], train.loc[:, column]
        model.fit(X, y)
        
        #print(model.score(X, y))
        mask = tmp[(tmp==1) & (data[column].isna())].index
      
        data_to_change = data.loc[mask]
        
        X = data_to_change.loc[:, cols]
        
        if len(X) == 0:
            return data
        #print(X)
        #print(model.predict(X))
        data.loc[mask, column] = model.predict(X)
        
                
        return data
                

        
    def string_to_date(self,year, month, day):
        
        # Russian month name to numerical mapping
        months = {
            'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
            'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
        }
        month = months[month]

        # Create a datetime object
        date = datetime( int(year), month, int(day) )

        # Truncate the datetime object to exclude time
        truncated_date = date.date()
        
        return truncated_date
        
    def excel_to_df(self,file_path, skip_rows=0):
        data = pd.read_excel(file_path, index_col='Скв.')
        data = data.iloc[skip_rows:]
        indices = data.index
        indices = indices[~indices.isna()]
        shit_words = [k for (k,v) in Counter(indices).items() if v > 1]

        indices = [i if i not in shit_words else np.nan for i in data.index]
        data.index = indices
        data.index = pd.Series(data.index).fillna(method='ffill')
        data.drop(columns = ['Unnamed: 1', 'Unnamed: 2'], inplace=True)
        
        columns1 = data.columns
        columns2 = list(data.iloc[0])
        years = [str(i) for i in range(2000, 2040)]
        
        
        columns = ['Состояние']
        
        for i in range(len(columns), len(columns1)):
            tmp = columns1[i].split()
            if len(tmp)==2:
                if tmp[1] in years:
                    month, year = tmp
            if 'МЭР' in str(columns2[i]) or 'ТР' in str(columns2[i]):
                columns.append(columns2[i]+' '+year)
            elif 'период' in columns1[i]:
                columns.append(columns1[i])
            else:
                columns.append( self.string_to_date( year, month, columns2[i] ) )
        columns[1] += f' {year}'    
        columns[-1] += f' {year}' 
        data.columns = columns
        return data
        
    def preprocess(self, snapshot):
        #prep = dict()
        
        snapshot = snapshot.iloc[1:]
        snapshot.set_index(keys='Состояние', drop=True, inplace=True)
        
        return snapshot

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.n:
            index = self.indices[self.index]
            print(index)
            value = self.data.loc[index]
            value = self.preprocess(value)
            self.index += 1
            return value
        else:
            self.index = 0
            raise StopIteration
    
    def __getitem__(self, index):
        if type(index) == int:
            if index < self.n:
                index = self.indices[index]
                value = self.data.loc[index]
                res = self.preprocess(value) 
            else:
                raise IndexError("Index out of range") 
        elif type(index) == str:
            if index in self.indices:
                value = self.data.loc[index]
                res = self.preprocess(value)
            elif index in self.mask_indices:
                ind = self.mask_indices.index(index)
                index = self.indices[ind]
                value = self.data.loc[index]
                res = self.preprocess(value) 
            else:
                raise IndexError("No such index!")     
                
        if self.mode == 'dynamic':
            return res
        elif self.mode == 'static':            
            Q = res.loc['Qж']
            W = res.loc['Обв']
            G = res.loc['Qгаз ТМ']/res.loc['Qн'].replace(0, np.inf)
            Pt = res.loc['Рбуф']
            Pb = res.loc['Рприем'].fillna(res.loc['Рэцн ТМ'])
            static_res = pd.DataFrame(columns=['Q', 'W', 'G', 'Pt', 'Pb'], data=np.array([Q,W,G,Pt,Pb]).T)
            static_res = static_res.reset_index(drop=True)
            if self.filtering:
                static_res = filter_data(static_res)#.drop_duplicates()
            
            if self.interpolate is not None:
                return self.interpolate_data(data=static_res, column=self.interpolate)
            else:
                return static_res
        else:
            raise Exception('Unknown mode! Please, choose "static" or "dynamic"')
            
            
    def full_info(self):
        data = self[0]
        for i in range(1, self.n):
            data = pd.concat([data, self[i]], axis=0, ignore_index=True)
        return data