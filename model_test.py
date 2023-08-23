import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import kaleido
import plotly.io as pio
import warnings
pio.renderers.default = 'png'
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")
##


def generate_grid():
    model_grid = pd.DataFrame(columns=['G', 'Q', 'W', 'Pt'], dtype='float32')
    index_count = 0
    g, g0 = 0, 0
    q, q0 = 1, 1
    w, w0 = 0, 0
    pt, pt0 = 1, 1

    g_step = 1000 / 10
    q_step = (1000 - 1) / 10
    w_step = 100 / 10
    pt_step = (20 - 1) / 10

    while g <= 1000:
        q = q0
        while q <= 1000:
            w = w0
            while w <= 100:
                pt = pt0
                while pt <= 20:
                    params = [g, q, w, pt]
                    model_grid.loc[index_count, :] = params
                    pt += pt_step
                    index_count += 1
                w += w_step
            q += q_step
        g += g_step
    extr_g_grid = model_grid.copy()
    for i in range(0, len(extr_g_grid)):
        extr_g_grid.iat[i, 0] = 10000

    stacked_df = pd.concat([model_grid, extr_g_grid], ignore_index=True)
    return stacked_df


# model_path = '/Users/stanislavananyev/PycharmProjects/GPN/models/modelsLug/New_no_inter_model'


def test_model(grid_df, model_path, indexes, err_list_save_path):
    counter = 0
    full_err_list = []
    for j in tqdm(indexes):
        loaded_model = pickle.load(open(model_path + '{}.pickle'.format(j), "rb"))
        pred_df = pd.DataFrame(columns=['Pt', 'Pb'], dtype='float32')
        for i in range(0, len(grid_df)):
            # params = grid_df.loc[i, :]
            # pred_df.loc[i, 'Pt'] = params[3]
            pred_df.loc[i, 'Pt'] = grid_df.loc[i, 'Pt']
            pred_df.loc[i, 'Pb'] = loaded_model.predict(grid_df.loc[i, :])
            # pred_df.loc[i, 'Pb'] = loaded_model.predict(params)

        # err_list = []
        for i in range(0, len(pred_df)):
            if pred_df.loc[i, 'Pt'] > pred_df.loc[i, 'Pb'] or pred_df.loc[i, 'Pb'] <= 0:
                full_err_list.append(j)
                break
        # full_err_list.append(err_list)
        np.savetxt(err_list_save_path, full_err_list)
    return full_err_list


def check_the_model(model_number, model_path, grid_df):
    pred_df = pd.DataFrame(columns=['Pb'], dtype='float32')
    full_pred_df = grid_df.copy()
    loaded_model = pickle.load(
        open(model_path + '/no_inter_model{}.pickle'.format(model_number), "rb"))
    for i in range(0, len(grid_df)):
        pred_df.loc[i, 'Pb'] = loaded_model.predict(grid_df.loc[i, :])
    full_pred_df = pd.concat((full_pred_df, pred_df['Pb']), axis=1)
    return full_pred_df


def plot_results(prediction_df, err_num, dict_path, image_path):
    dictionary = pickle.load(
        open(dict_path, "rb"))
    well_name = list(dictionary.keys())[list(dictionary.values()).index(err_num)]

    yx_line = prediction_df.copy()
    for i in range(0, len(yx_line)):
        yx_line.loc[i, 'Pt'] = i
    yx_line['Pb'] = yx_line['Pt']
    axis_count = 0
    min_value = 0
    max_value = 0
    for i in range(0, len(prediction_df)):
        for j in range(0, 120):
            if prediction_df.loc[120*i+j, 'Pb'] < min_value:
                min_value = prediction_df.loc[120*i+j, 'Pb']
            if prediction_df.loc[120 * i + j, 'Pb'] > max_value:
                max_value = prediction_df.loc[120 * i + j, 'Pb']
            if prediction_df.loc[120*i+j, 'Pb'] <= 0 or prediction_df.loc[120*i+j, 'Pt'] >= prediction_df.loc[120*i+j, 'Pb']:
                axis_count += 1
                break
        if axis_count < i+1 or axis_count >= 9:
            break
    fig = px.scatter(title='{}'.format(well_name))
    fig.update_layout(xaxis_title="Pt", yaxis_title="Pb", font_size=15)
    fig.update_layout(title_pad_l=500, title_font_size=30)
    # fig.update_layout(xaxis_range=[0, 20], yaxis_range=[-50, 150], xaxis_title="Pt", yaxis_title="Pb")
    fig.update_layout(xaxis_range=[0, 20], yaxis_range=[min_value - 50, max_value+50], xaxis_title="Pt", yaxis_title="Pb")
    for i in range(0, axis_count+2):
        temp = 1 if i == 0 else 0
        fig.add_trace(go.Scatter(x=prediction_df['Pt'][i*120+1-temp:i*120+121],
                                 y=prediction_df['Pb'][i*120+1-temp:i*120+121],
                                 mode='markers', name="[G=0; Q={}; W:(0,100); Pt:(1,20)]".format(100*i+temp)))

    fig.add_trace(go.Scatter(x=yx_line['Pt'], y=yx_line['Pb'], name="Pb=Pt", line=dict(color="black")))
    fig.update_layout(width=1000, height=800)

    # fig.update_layout(bargap=0.2)
    # fig.show()
    fig.write_image(image_path + "/fig{}.png".format(err_num), scale=3)

# dictionary_path = '/Users/stanislavananyev/PycharmProjects/GPN/well_dict.pickle'
#
# grid = generate_grid()
# df = check_the_model(46, grid)
# plot_results(df, 46, dictionary_path)
# ##
# dictionary_path = '/Users/stanislavananyev/PycharmProjects/GPN/well_dict.pickle'
# dict_path = dictionary_path
# dictionary = pickle.load(
#     open(dict_path, "rb"))
##

