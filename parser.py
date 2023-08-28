import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import generator
import model_test
from generator import Snapshot_Generator
import random
import pickle
from model import super_model, my_custom_scorer
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# data_path = '/Users/stanislavananyev/PycharmProjects/GPN/Wells_data.xlsx'


# def filter_data(data):
#     err = 0
#     cond1 = (data['ГФ'] <= 3000)
#     cond2 = (data['Qж'] <= 200)
#     cond3 = (data['Рприем'] <= 200)
#     cond4 = (data['Обв'] <= 100)
#     cond5 = (data['Рбуф'] <= 200)
#     cond6 = (data['Рбуф'] <= data['Рприем'])
#     if cond1 is False or cond2 is False or cond3 is False or \
#             cond4 is False or cond5 is False or cond6 is False:
#         err = 1
#     return err
def filter_data(data):
    err = 0
    if data['ГФ'] > 3000 or data['Qж'] > 200 or data['Рприем'] > 200 or \
            data['Обв'] > 100 or data['Рбуф'] > 200 or data['Рбуф'] > data['Рприем']:
        err = 1
    return err


def filter_after_inter(data):
    err = 0
    if data['G'] > 3000 or data['Q'] > 200 or data['Pb'] > 200 or \
            data['W'] > 100 or data['Pt'] > 200 or data['Pt'] > data['Pb']:
        err = 1
    return err


def build_data_df(excel_path, index_path, drop_zero_g):
    wells_df = pd.read_excel(excel_path, skiprows=[0, 1, 2], header=None)
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
            # if filter_data(train_df.iloc[i]) != 0 or train_df.loc[i, 'ГФ'] == 0:
            if filter_data(train_df.iloc[i]) != 0:
                filter_drop_list.append(i)
            elif drop_zero_g is True and train_df.loc[i, 'ГФ'] == 0:
                filter_drop_list.append(i)
        train_df.drop(['Qн', 'Qгаз ТМ'], axis=1, inplace=True)
        train_df.drop(train_df.index[filter_drop_list], axis=0, inplace=True)
        train_df = train_df[['ГФ', 'Qж', 'Обв', 'Рбуф', 'Рприем']]
        train_df.rename(columns={'ГФ': 'G', 'Qж': 'Q', 'Обв': 'W', 'Рбуф': 'Pt', 'Рприем': 'Pb'}, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        if not train_df.shape[0] < 50:
            df_list.append(train_df)
            wells_index_list.append(j+1)
    np.savetxt(index_path + '/wells_index.txt', wells_index_list, fmt='%d')
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


def train_one_model(data, model_path, model_index):
    test_size = int(data.shape[0]*0.15)
    temp_data = data.sample(frac=1)
    train = data.iloc[:data.shape[0] - test_size, :]
    test = data.iloc[data.shape[0] - test_size:, :]

    model = super_model()
    model.fit(train)
    filename = model_path + '/{}.pickle'.format(model_index)
    pickle.dump(model, open(filename, "wb"))
    x, y = test.iloc[:, :-1], test.iloc[:, -1]
    score = model.evaluate(x, y)
    return score


# def interpolate_data(data_list):
#     for i in range(0, len(data_list)):
#         for j in range(0, len(data_list[i])):
#             if data_list[i].loc[j, 'G'] == 0:
#                 data_list[i].loc[j, 'G'] = np.nan
#         imp = IterativeImputer(max_iter=10, min_value=10, initial_strategy='median')
#         data = imp.fit_transform(data_list[i])
#         data_list[i] = pd.DataFrame(data)
#     return data_list
def interpolate_data(data_list):
    for i in range(0, len(data_list)):
        for j in range(0, len(data_list[i])):
            if data_list[i].loc[j, 'G'] == 0:
                data_list[i].loc[j, 'G'] = np.nan
        imp = IterativeImputer(max_iter=10, min_value=10, initial_strategy='median')
        data = imp.fit_transform(data_list[i])
        data_list[i] = pd.DataFrame(data, columns=['G', 'Q', 'W', 'Pt', 'Pb'])
    filter_drop_list = []
    for i in range(0, len(data_list)):
        for j in range(0, len(data_list[i])):
            if filter_after_inter(data_list[i].iloc[j]) != 0:
                filter_drop_list.append(j)
        data_list[i].drop(data_list[i].index[filter_drop_list], axis=0, inplace=True)
    return data_list


