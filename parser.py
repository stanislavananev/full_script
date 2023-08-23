
import keras
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import generator
from generator import Snapshot_Generator
import tensorflow as tf
import random

# data_path = '/Users/stanislavananyev/PycharmProjects/GPN/Wells_data.xlsx'


def filter_data(data):
    err = 0
    cond1 = (data['ГФ'] <= 3000)
    cond2 = (data['Qж'] <= 200)
    cond3 = (data['Рприем'] <= 200)
    cond4 = (data['Обв'] <= 100)
    cond5 = (data['Рбуф'] <= 200)
    cond6 = (data['Рбуф'] <= data['Рприем'])
    if cond1 is False or cond2 is False or cond3 is False or \
            cond4 is False or cond5 is False or cond6 is False:
        err = 1
    return err


def build_data_df(path):
    wells_df = pd.read_excel(path, skiprows=[0, 1, 2], header=None)
    wells_df.drop(columns=[0, 1, 2, 4, 5, 6, 8, 9], inplace=True)
    wells_df.drop(index=[0], axis=0, inplace=True)
    wells_index_list = []
    arr_list = []
    for j in range(0, wells_df.shape[0], 7):
        df1 = wells_df.copy()[j:j+7]
        df1.dropna(axis=1, inplace=True, how='all')
        df1.fillna(value="0", inplace=True)
        trash_list = []
        for i in range(2, df1.shape[1]):
            if df1.iloc[0, i] == "0" or df1.iloc[1, i] == "0" or df1.iloc[3, i] == "0" or df1.iloc[4, i] == "0":
                trash_list.append(i)
        df1.drop(df1.columns[trash_list], axis=1, inplace=True)
        arr_list.append(np.array(df1))

    inter_ind = []
    df_list = []

    for j in range(0, len(arr_list)):
        train_df = pd.DataFrame(arr_list[j].T)
        train_df.columns = train_df.iloc[1]
        train_df.drop(index=[0, 1], inplace=True)
        train_df.drop_duplicates(inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        filter_drop_list = []
        for column in train_df:
            train_df[column] = [x.replace(',', '.') for x in train_df[column]]
            train_df[column] = train_df[column].astype('float32')
        for i in range(0, len(train_df)):
            if train_df.loc[i, 'Qгаз ТМ'] != 0 and train_df.loc[i, 'Qн'] != 0:
                train_df.loc[i, 'ГФ'] = train_df.loc[i, 'Qгаз ТМ']/train_df.loc[i, 'Qн']
            elif train_df.loc[i, 'Qгаз ТМ'] != 0 and train_df.loc[i, 'Qн'] == 0:
                train_df.loc[i, 'ГФ'] = 1.0
            if filter_data(train_df.iloc[i]) != 0 or train_df.loc[i, 'ГФ'] == 0:
                filter_drop_list.append(i)
        train_df.drop(['Qн', 'Qгаз ТМ'], axis=1, inplace=True)
        train_df.drop(train_df.index[filter_drop_list], axis=0, inplace=True)
        train_df = train_df[['ГФ', 'Qж', 'Обв', 'Рбуф', 'Рприем']]
        train_df.rename(columns={'ГФ': 'G', 'Qж': 'Q', 'Обв': 'W', 'Рбуф': 'Pt', 'Рприем': 'Pb'}, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        if not train_df.shape[0] < 100:
            df_list.append(train_df)
            wells_index_list.append(j+1)
    np.savetxt('/Users/stanislavananyev/PycharmProjects/GPN/wells_index.txt', wells_index_list, fmt='%d')
    return df_list, wells_index_list


def prepare_data(data_df):
    full_train_list = []
    full_valid_list = []
    train_df = []
    valid_df = []
    for i in range(0, len(data_df)):
        train_count = []
        valid_count = []
        for j in range(0, len(data_df[i])):
            train_count.append(j)
        valid_count = random.sample(train_count, int(len(train_count)*0.14))
        train_count = list(set(train_count)-set(valid_count))
        full_train_list.append(train_count)
        full_valid_list.append(valid_count)
        train_data = data_df[i].copy().drop(data_df[i].index[full_valid_list[i]])
        valid_data = data_df[i].copy().drop(data_df[i].index[full_train_list[i]])
        train_df.append(train_data)
        valid_df.append(valid_data)
    return train_df, valid_df


trained_model_path = '/Users/stanislavananyev/PycharmProjects/GPN/models/modelsLug'


def train_model(train_data, valid_data, model_path, model_indexes):
    # Тренировка моделей####
    import pickle
    from tqdm import tqdm
    from model import super_model, my_custom_scorer
    eval_df = pd.DataFrame(columns=['Score'])
    counter = 0

    for i in tqdm(model_indexes):
        train = train_data[counter].copy()
        valid = valid_data[counter].copy()
        model = super_model()
        model.fit(train)
        filename = model_path + '/New_no_inter_model{}.pickle'.format(i)
        pickle.dump(model, open(filename, "wb"))
        x, y = valid.iloc[:, :-1], valid.iloc[:, -1]
        score = model.evaluate(x, y)
        eval_df.loc[i, 'Score'] = score
        counter += 1
    return eval_df

