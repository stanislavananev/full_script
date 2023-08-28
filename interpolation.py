import numpy as np

import parser
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from model import super_model, my_custom_scorer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def pause():
    program_pause = input("Press the <ENTER> key to continue...")


data_df = parser.build_data_df('/Users/stanislavananyev/PycharmProjects/GPN/Wells_data.xlsx',
                               '/Users/stanislavananyev/PycharmProjects/GPN/temp_folder', drop_zero_g=False)[0]

stacked_data = data_df[0].copy()
for i in range(1, len(data_df)):
    stacked_data = pd.concat((stacked_data, data_df.copy()[i]))
stacked_data.reset_index(drop=True, inplace=True)
# stacked_data = stacked_data.sample(frac=1, random_state=1)
valid_size = int(stacked_data.shape[0]*0.2)
train_x = stacked_data.copy().iloc[:stacked_data.shape[0] - valid_size, :]
train_y = stacked_data.copy().iloc[:stacked_data.shape[0] - valid_size, :]
valid_x = stacked_data.copy().iloc[stacked_data.shape[0] - valid_size:, :]
valid_y = stacked_data.copy().iloc[stacked_data.shape[0] - valid_size:, :]

for i in range(0, len(stacked_data)):
    if stacked_data.loc[i, 'G'] == 0:
        stacked_data.loc[i, 'G'] = np.nan

imp = IterativeImputer(max_iter=10)
data = imp.fit_transform(stacked_data)
stacked_data = pd.DataFrame(data)
# print(stacked_data)
# new_df = imp.transform(valid_x)
# model = DecisionTreeRegressor(max_depth=500, random_state=0)
# model.fit(train_x, train_y)
#
# data_to_interpolate = parser.build_data_df('/Users/stanislavananyev/PycharmProjects/GPN/Wells_data.xlsx',
#                                            '/Users/stanislavananyev/PycharmProjects/GPN/temp_folder',
#                                            drop_zero_g=False)[0]
# print()

